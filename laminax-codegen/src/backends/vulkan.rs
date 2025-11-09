//! Vulkan backend for cross-platform GPU compute.

use crate::CodegenError;
use crate::backends::{Backend, BackendCapabilities};
use std::sync::Arc;
use vulkano::device::physical::PhysicalDevice;
use vulkano::instance::debug::{
    ValidationFeatureEnable,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::{Version, VulkanLibrary};
use vulkano::device::{Queue, QueueCreateInfo, QueueFlags};

/// Vulkan backend implementation
pub struct VulkanBackend {
    device: Arc<vulkano::device::Device>,
    device_name: String,
    queues: Vec<Arc<Queue>>,
}

struct VerificationReport {
    compute_queue_index: u32,
}

const IS_DEBUG: bool = cfg!(debug_assertions);
const VALIDATOR: &str = "VK_LAYER_KHRONOS_validation";
const MIN_MAC_VERSION: Version = Version {
    major: 1,
    minor: 3,
    patch: 216,
};


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
        )
            .expect("Unable to create Vulkan Instance!");

        let devices: Vec<_> = instance
            .enumerate_physical_devices()
            .expect("Unable to enumerate Vulkan Physical Devices!")
            .collect();
        let verification_results: Vec<_> = devices
            .iter()
            .cloned()
            .map(Self::verify_device)
            .collect();

        let mut compute_queue_index = u32::MAX;
        let mut best_match = None;
        for (result, device) in verification_results.iter().zip(devices.iter()) {
            if let Ok(report) = result {
                compute_queue_index = report.compute_queue_index;
                best_match = Some(device.clone());
                break;
            }
        }

        let best_match = match best_match {
            Some(v) => v,
            None => {
                eprintln!("No suitable device found!");
                for (message, device) in verification_results.into_iter().zip(devices.into_iter()) {
                    let Err(message) = message else { unreachable!() };
                    eprintln!("Device name: {} Reason: {}", device.properties().device_name, message);
                }
                panic!();
            }
        };
        let device_name = best_match.properties().device_name.clone();

        let (device, queues) = vulkano::device::Device::new(best_match, vulkano::device::DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: compute_queue_index,
                ..Default::default()
            }],
            ..vulkano::device::DeviceCreateInfo::default()
        }).expect("Unable to create Vulkan Device!");
        let queues: Vec<_> = queues.collect();

        Self {
            device,
            device_name,
            queues,
        }
    }

    /// Compile SPIR-V shader
    pub fn compile_spirv(
        &self,
        _spirv_bytes: &[u8],
    ) -> Result<Vec<u8>, CodegenError> {
        Err(CodegenError::NotImplemented(
            "SPIR-V compilation not yet implemented",
        ))
    }

    /// Compile from LCIR to SPIR-V
    pub fn compile_from_lcir(
        &self,
        _kernel: &laminax_lcir::Kernel,
    ) -> Result<Vec<u8>, CodegenError> {
        let spirv = crate::lowering::spirv::lower_lcir_to_spirv(_kernel)?;
        self.compile_spirv(spirv.as_bytes())
    }

    fn verify_device(device: Arc<PhysicalDevice>) -> Result<VerificationReport, String> {
        let queue_properties = device.queue_family_properties();

        let Some(compute_queue_index) = queue_properties.iter().position(|queue| queue.queue_flags.contains(QueueFlags::COMPUTE)) else {
            return Err("This device does not support compute shader".to_string());
        };
        let compute_queue_index = compute_queue_index.try_into().expect("Invalid Compute Queue Index");

        Ok(VerificationReport {
            compute_queue_index,
        })
    }
}

impl Backend for VulkanBackend {
    fn capabilities(&self) -> BackendCapabilities {
        let features = self.device.enabled_features();

        let supports_fp64 = features.shader_float64;
        let supports_fp16 = features.shader_float16;
        let supports_int64 = features.shader_int64;
        let supports_int16 = features.shader_int16;
        let supports_int8 = features.shader_int8;
        let supports_async = true;
        let unified_memory = false;
        let shared_memory = true;


        BackendCapabilities {
            supports_fp64,
            supports_fp16,
            supports_int64,
            supports_int16,
            supports_int8,
            supports_async,
            unified_memory,
            shared_memory,
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Vulkan"
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        VulkanBackend::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::Backend;

    #[test]
    fn test_instance_creation() {
        use crate::backends::vulkan::VulkanBackend;
        let backend = VulkanBackend::new();
        println!("Name: {}", &backend.device_name);
        println!("Capabilities: {:?}", backend.capabilities());
    }
}
