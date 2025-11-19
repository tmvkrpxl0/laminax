//! Vulkan backend for cross-platform GPU compute.

use crate::memory::{BufferProperties, DeviceBuffer};
use crate::{Backend, Buffer, Device, DeviceCapabilities, DeviceType, MemoryManager, RuntimeError};
use laminax_types::DType;
use laminax_types::Shape;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use vulkano::buffer::{AllocateBufferError, BufferContents, BufferCreateFlags, BufferCreateInfo, BufferUsage};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{DeviceFeatures, Queue, QueueCreateInfo, QueueFlags};
use vulkano::half::{bf16, f16};
use vulkano::instance::debug::ValidationFeatureEnable;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::sync::Sharing;
use vulkano::{DeviceSize, Validated, Version, VulkanLibrary};
use vulkano::buffer::Buffer as VkBuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, CopyBufferInfoTyped, PrimaryAutoCommandBuffer};

/// Vulkan Backend Instance holder
pub struct VulkanBackend {
    instance: Arc<Instance>,
}

/// Vulkan device implementation
#[derive(Debug)]
pub struct VulkanDevice {
    device: Arc<vulkano::device::Device>,
    device_name: String,
    queues: Vec<Arc<Queue>>,
    capabilities: DeviceCapabilities,
    allocator: Arc<dyn MemoryAllocator>,
    id: AtomicUsize,
    staging_records: Option<RwLock<Vec<(Vec<u8>, Arc<VkBuffer>)>>>,
}

struct VerificationReport {
    compute_queue_index: u32,
    needs_staging: bool,
}


const IS_DEBUG: bool = cfg!(debug_assertions);
const VALIDATOR: &str = "VK_LAYER_KHRONOS_validation";
const MIN_MAC_VERSION: Version = Version {
    major: 1,
    minor: 3,
    patch: 216,
};

impl Backend for VulkanBackend {
    type Device = VulkanDevice;
    const BACKEND_TYPE: DeviceType = DeviceType::Vulkan;

    fn discover_devices(&self, required_capabilities: DeviceCapabilities) -> Vec<Arc<Self::Device>> {
        let devices: Vec<_> = self.instance
            .enumerate_physical_devices()
            .expect("Unable to enumerate Vulkan Physical Devices!")
            .collect();
        let verification_results: Vec<_> = devices
            .iter()
            .map(|v| Self::verify_device(v.clone(), required_capabilities))
            .collect();
        debug_assert_eq!(verification_results.len(), devices.len());

        let available: Vec<_> = verification_results
            .iter().zip(devices.iter()).filter_map(|(result, device)| {
            if let Ok(report) = result {
                Some((device.clone(), report))
            } else {
                None
            }
        }).map(|(device, report)| {
            VulkanDevice::new(device, report.compute_queue_index, required_capabilities, report.needs_staging)
        }).collect();

        if available.is_empty() {
            eprintln!("No suitable device found!");
            for (message, device) in verification_results.into_iter().zip(devices.into_iter()) {
                let Err(message) = message else { unreachable!() };
                eprintln!("Device name: {} Reason: {}", device.properties().device_name, message);
            }
            panic!();
        }

        available
    }
}

impl VulkanBackend {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().expect("Error loading Vulkan");

        let mut enabled_layers = vec![];
        let mut enabled_validation_features = vec![];

        let mut enabled_extensions = InstanceExtensions {
            ext_debug_utils: IS_DEBUG,
            ..InstanceExtensions::default()
        };

        if IS_DEBUG
            && let Ok(mut properties) = library.layer_properties().inspect_err(|e| {
            eprintln!(
                "Debug Mode detected but Vulkan Validation Layer cannot be used!: {:?}",
                e
            )
        })
            && let Some(_) = properties.find(|layer| layer.name() == VALIDATOR)
        {
            enabled_layers.push(VALIDATOR.to_string());
            enabled_validation_features.push(ValidationFeatureEnable::DebugPrintf);
            println!("Validation Layer Enabled!");
            enabled_extensions.ext_validation_features = true;
        };

        let flags = if cfg!(target_os = "macos") && library.api_version() >= MIN_MAC_VERSION {
            println!("Mac Compatibility: Activated");
            enabled_extensions.khr_get_physical_device_properties2 = true;
            enabled_extensions.khr_portability_enumeration = true;
            InstanceCreateFlags::ENUMERATE_PORTABILITY
        } else {
            InstanceCreateFlags::empty()
        };

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags,
                engine_name: Some("Vulkan Backend".to_string()),
                engine_version: Version::major_minor(0, 1),
                enabled_layers,
                enabled_validation_features,
                enabled_extensions,
                ..InstanceCreateInfo::application_from_cargo_toml()
            },
        ).expect("Unable to create Vulkan Instance!");

        Self {
            instance,
        }
    }

    fn verify_device(device: Arc<PhysicalDevice>, required_capabilities: DeviceCapabilities) -> Result<VerificationReport, String> {
        let queue_properties = device.queue_family_properties();

        let Some(compute_queue_index) = queue_properties.iter().position(|queue| queue.queue_flags.contains(QueueFlags::COMPUTE)) else {
            return Err("This device does not support compute shader".to_string());
        };
        let compute_queue_index = compute_queue_index.try_into().expect("Invalid Compute Queue Index");

        let DeviceCapabilities {
            compute_units,
            max_work_group_size,
            local_memory_size,
            global_memory_size,
            supports_fp64,
            supports_fp16,
            supports_async: _supports_async,
            unified_memory,
            shared_memory
        } = required_capabilities;

        let device_properties = device.properties();
        let memory_properties = device.memory_properties();
        let supported_features = device.supported_features();
        if (device_properties.max_compute_work_group_invocations as usize) < compute_units {
            return Err(format!(
                "This device only supports work group invocations up to {}. Required: {}",
                device_properties.max_compute_work_group_invocations,
                required_capabilities.max_work_group_size
            ));
        }

        let device_max_work_group_size = device_properties.max_work_group_size.expect(
            "Maximum Workgroup size is not reported for this device,\
                 despite reporting it's capable of compute shader."
        );
        if (device_max_work_group_size
            .iter()
            .cloned()
            .product::<u32>() as usize) < max_work_group_size {
            return Err(format!(
                "This device only supports work group size up to [{}, {}, {}]. Required: {}",
                device_max_work_group_size[0],
                device_max_work_group_size[1],
                device_max_work_group_size[2],
                max_work_group_size
            ));
        }

        if (device_properties.max_compute_shared_memory_size as usize) < local_memory_size {
            return Err(format!(
                "This device only supports work group size up to {}. Required: {}",
                device_properties.max_compute_shared_memory_size,
                local_memory_size
            ));
        }

        let max_memory_size = device_properties.max_memory_allocation_size.unwrap_or_else(|| {
            memory_properties.memory_heaps.iter().map(|v| v.size).sum()
        }) as usize;

        if max_memory_size < global_memory_size {
            return Err(format!(
                "This device does not have enough memory. Reported: {} bytes. Required: {} bytes",
                max_memory_size, global_memory_size
            ));
        };

        if !(supports_fp64 && supported_features.shader_float64) {
            return Err("This device does not support 64-bit float representation!".to_string());
        }

        if !(supports_fp16 && supported_features.shader_float16) {
            return Err("This device does not support 16-bit float representation!".to_string());
        }

        if !(unified_memory && !memory_properties
            .memory_types
            .iter()
            .any(|v|
                v.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE) &&
                    !v.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
            )) {
            return Err("This device does not support unified memory!".to_string());
        }

        if !(shared_memory && !memory_properties
            .memory_types
            .iter()
            .any(|v|
                v.property_flags.contains(
                    MemoryPropertyFlags::HOST_VISIBLE |
                        MemoryPropertyFlags::DEVICE_LOCAL
                )
            )) {
            return Err("This device does not support sharing memory with host!".to_string());
        }

        let needs_staging = memory_properties
            .memory_types
            .iter()
            .any(|v| {
                let local = v.property_flags.contains(
                    MemoryPropertyFlags::DEVICE_LOCAL
                );
                let invisible = !v.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE);
                local && invisible
            });

        Ok(VerificationReport {
            compute_queue_index,
            needs_staging,
        })
    }
}

impl VulkanDevice {
    pub fn new(device: Arc<PhysicalDevice>, compute_queue_index: u32, required_capabilities: DeviceCapabilities, allocate_staging: bool) -> Arc<Self> {
        let device_name = device.properties().device_name.clone();

        let enabled_features = DeviceFeatures {
            shader_float64: required_capabilities.supports_fp64,
            shader_float16: required_capabilities.supports_fp16,
            ..DeviceFeatures::default()
        };

        let (device, queues) = vulkano::device::Device::new(device, vulkano::device::DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: compute_queue_index,
                ..Default::default()
            }],
            enabled_features,
            ..vulkano::device::DeviceCreateInfo::default()
        }).expect("Unable to create Vulkan Device!");
        let queues: Vec<_> = queues.collect();
        let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let staging_buffer = if allocate_staging {
            Some(RwLock::new(vec![]))
        } else {
            None
        };

        Arc::new(Self {
            device,
            device_name,
            queues,
            capabilities: required_capabilities,
            allocator,
            id: AtomicUsize::new(0),
            staging_records: staging_buffer,
        })
    }
}

impl Device for VulkanDevice {
    type Backend = VulkanBackend;

    fn device_type() -> DeviceType {
        DeviceType::Vulkan
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Vulkan"
    }
}

impl MemoryManager for VulkanDevice {
    type Device = Self;
    type Storage = VkBuffer;
    type FlushTarget = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

    fn allocate(self: Arc<Self>, shape: Shape, dtype: DType, properties: BufferProperties) -> crate::Result<crate::Buffer<VkBuffer>> {
        let element_layout = match dtype {
            DType::F32 => <f32 as BufferContents>::LAYOUT,
            DType::F64 => <f64 as BufferContents>::LAYOUT,
            DType::F16 => <f16 as BufferContents>::LAYOUT,
            DType::I8 => <i8 as BufferContents>::LAYOUT,
            DType::I16 => <i16 as BufferContents>::LAYOUT,
            DType::I32 => <i32 as BufferContents>::LAYOUT,
            DType::I64 => <i64 as BufferContents>::LAYOUT,
            DType::U8 => <u8 as BufferContents>::LAYOUT,
            DType::U16 => <u16 as BufferContents>::LAYOUT,
            DType::U32 => <u32 as BufferContents>::LAYOUT,
            DType::U64 => <u64 as BufferContents>::LAYOUT,
            DType::Bool => <u8 as BufferContents>::LAYOUT,
            DType::BF16 => <bf16 as BufferContents>::LAYOUT,
            DType::QI4 => return Err(RuntimeError::Device("4 bit types are not supported".to_string())),
            DType::QU8 => <u8 as BufferContents>::LAYOUT,
        };
        let BufferProperties {
            copy_source,
            copy_destination,
            device_local,
            device_write,
            host_visible,
        } = properties;
        let usage = {
            let mut usage = BufferUsage::empty();

            if copy_source.unwrap_or(false) {
                usage |= BufferUsage::TRANSFER_SRC;
            }
            if copy_destination.unwrap_or(false) {
                usage |= BufferUsage::TRANSFER_DST;
            }
            if device_write.unwrap_or(false) {
                usage |= BufferUsage::STORAGE_BUFFER;
            }
            usage
        };

        let required_memory_flags = {
            let mut flags = MemoryPropertyFlags::empty();
            if device_local.unwrap_or(false) {
                flags |= MemoryPropertyFlags::DEVICE_LOCAL;
            };
            if host_visible.unwrap_or(false) {
                flags |= MemoryPropertyFlags::HOST_VISIBLE;
            }
            flags
        };
        let allocated = VkBuffer::new(
            self.allocator.clone(),
            BufferCreateInfo {
                // It seems possible to support sparse layout in BufferCreateFlags.
                // Not sure how that'd be useful for our purposes but noting here in case someone wants it.
                flags: BufferCreateFlags::default(),

                // This should be chanced if we were to ever support multi-gpu support.
                sharing: Sharing::Exclusive,
                usage,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter {
                    required_flags: required_memory_flags,
                    ..MemoryTypeFilter::default()
                },
                ..AllocationCreateInfo::default()
            },
            element_layout.layout_for_len(shape.len() as DeviceSize)
                .ok_or(RuntimeError::Memory(format!("Invalid shape for tensor: {}", shape)))?
        )?;
        let id = self.id.fetch_add(1, Ordering::Relaxed);
        Ok(crate::Buffer {
            id,
            shape,
            dtype,
            device: self.clone(),
            data: allocated,
        })
    }

    fn deallocate(self: Arc<Self>, buffer: &crate::Buffer<VkBuffer>) -> crate::Result<()> {
        // Implemented in Drop trait
        Ok(())
    }

    fn copy(self: Arc<Self>, src: &crate::Buffer<VkBuffer>, dst: &crate::Buffer<VkBuffer>) -> crate::Result<()> {
        todo!()
    }

    fn require_staging(&self) -> bool {
        true
    }

    fn stage(&self, data: Vec<u8>, target: &Buffer<Self::Storage>) -> crate::Result<()> {
        let Some(staging_records) = self.staging_records.as_ref() else {
            return Err(RuntimeError::Device("This device does not require staging records".to_string()));
        };
        let is_supported = match target.dtype {
            DType::I8 | DType::U8 | DType::QU8 => true,
            _ => false,
        };
        if is_supported {
            return Err(RuntimeError::Device("Only 8 bit data types are supported for now".to_string()));
        }

        let mut lock = staging_records.write().expect("Internal Staging Record is Poisoned!");
        let size = size_of_val(data.as_slice());
        let mut new_buffer: Vec<u8> = Vec::with_capacity(size);

        new_buffer.copy_from_slice(data.as_slice());
        lock.push((new_buffer, target.data.clone()));
        Ok(())
    }

    fn flush(&self, flush_target: &mut Self::FlushTarget) -> crate::Result<()> {
        let Some(staging_records) = self.staging_records.as_ref() else {
            return Err(RuntimeError::Device("This device does not require staging records".to_string()));
        };

        let mut lock = staging_records.write().expect("Internal Staging Record is Poisoned!");
        for (record, storage) in lock.drain(..) {
            flush_target.copy_buffer(CopyBufferInfoTyped {
                ..CopyBufferInfoTyped::buffers(todo!(), todo!())
            }).expect("Error Handle not implemented");
        }

        Ok(())
    }
}

impl DeviceBuffer for VkBuffer {
    type Backend = VulkanBackend;
}

impl Default for VulkanBackend {
    fn default() -> Self {
        VulkanBackend::new()
    }
}

impl<E: Into<RuntimeError>> From<Validated<E>> for RuntimeError {
    fn from(value: Validated<E>) -> Self {
        match value {
            Validated::Error(e) => e.into(),
            Validated::ValidationError(e) => {
                RuntimeError::Device(format!("{:?}", e.as_ref()))
            }
        }
    }
}

impl From<AllocateBufferError> for RuntimeError {
    fn from(value: AllocateBufferError) -> Self {
        RuntimeError::Memory(format!("{:?}", value))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Backend, Device, DeviceCapabilities};
    use crate::vulkan::{VulkanBackend, VulkanDevice};

    #[test]
    fn test_instance_creation() {
        let backend = VulkanBackend::new();
        let requirement = DeviceCapabilities {
            compute_units: 0,
            max_work_group_size: 0,
            local_memory_size: 0,
            global_memory_size: 0,
            supports_fp64: true,
            supports_fp16: false,
            supports_async: false,
            unified_memory: false,
            shared_memory: false,
        };
        let devices = backend.discover_devices(requirement);
        let device = devices.first().unwrap();
        println!("Name: {}", device.name());
        println!("Capabilities: {:?}", device.capabilities());
    }
}
