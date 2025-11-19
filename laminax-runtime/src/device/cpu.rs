//! Device abstraction layer for heterogeneous computing
//!
//! Provides unified interfaces for different compute devices (CPU, GPU, etc.)
//! Uses device types from laminax-types.

use crate::device::{Device, DeviceCapabilities, DeviceType};
use crate::memory::{BufferProperties, DeviceBuffer};
use crate::{Backend, Buffer, MemoryManager, RuntimeError};
use laminax_types::{DType, Shape};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// CPU device implementation
pub struct CpuDevice {
    capabilities: DeviceCapabilities,
    next_buffer_id: AtomicUsize,
}

/// Dummy Backend for CPU
pub struct CpuBackend;

impl CpuDevice {
    pub fn new() -> Self {
        let capabilities = DeviceCapabilities {
            compute_units: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1), // Fallback to 1 if unavailable
            max_work_group_size: 1024,           // Arbitrary limit for CPU
            local_memory_size: 32 * 1024 * 1024, // 32MB L1/L2 cache estimate
            global_memory_size: get_system_memory(),
            supports_fp64: true,
            supports_fp16: cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"),
            supports_async: false, // CPU operations are typically synchronous
            unified_memory: true,  // CPU memory is unified
            shared_memory: false,  // No special shared memory on CPU
        };

        Self {
            capabilities,
            next_buffer_id: AtomicUsize::new(0),
        }
    }
}

impl Backend for CpuBackend {
    type Device = CpuDevice;
    const BACKEND_TYPE: DeviceType = DeviceType::Cpu;

    fn discover_devices(&self, required_capabilities: DeviceCapabilities) -> Vec<Arc<Self::Device>> {
        // TODO Support multi cpu
        let cpu = CpuDevice::new();
        if !cpu.capabilities.satisfies(&required_capabilities) {
            vec![]
        } else {
            vec![Arc::new(cpu)]
        }
    }
}

impl Device for CpuDevice {
    type Backend = CpuBackend;

    fn device_type() -> DeviceType {
        DeviceType::Cpu
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "CPU"
    }
}

impl MemoryManager for CpuDevice {
    type Device = Self;
    type Storage = Vec<u8>;
    type FlushTarget = ();

    fn allocate(
        self: Arc<Self>,
        shape: Shape,
        dtype: DType,
        _: BufferProperties,
    ) -> crate::Result<Buffer<Vec<u8>>> {
        let id = self
            .next_buffer_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // For CPU, allocate actual memory
        let size_bytes = shape.len() * dtype.dtype_size_bytes();
        let data = Arc::new(vec![0u8; size_bytes]);

        Ok(Buffer {
            id,
            shape,
            dtype,
            device: self.clone(),
            data,
        })
    }

    fn deallocate(self: Arc<Self>, _buffer: &Buffer<Vec<u8>>) -> crate::Result<()> {
        // Placeholder for deallocation
        Ok(())
    }

    fn copy(self: Arc<Self>, _src: &Buffer<Vec<u8>>, _dst: &Buffer<Vec<u8>>) -> crate::Result<()> {
        // Placeholder for memory copy operations
        Ok(())
    }

    fn require_staging(&self) -> bool {
        false
    }

    fn stage(&self, _: Vec<u8>, _: &Buffer<Self::Storage>) -> crate::Result<()> {
        Err(RuntimeError::Device("This device does not require staging records".to_string()))
    }

    fn flush(&self, _: &mut Self::FlushTarget) -> crate::Result<()> {
        Err(RuntimeError::Device("This device does not require staging records".to_string()))
    }
}

impl DeviceBuffer for Vec<u8> {
    type Backend = CpuBackend;
}

/// Get system memory using std::fs and /proc/meminfo (Linux) or other platform-specific methods
fn get_system_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Try reading /proc/meminfo on Linux
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Try using sysctl on macOS
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(mem_str) = std::str::from_utf8(&output.stdout) {
                if let Ok(mem) = mem_str.trim().parse::<usize>() {
                    return mem;
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Try using systeminfo on Windows
        use std::process::Command;
        if let Ok(output) = Command::new("wmic")
            .args(["ComputerSystem", "get", "TotalPhysicalMemory"])
            .output()
        {
            if let Ok(mem_str) = std::str::from_utf8(&output.stdout) {
                // Parse the output (skip header line)
                for line in mem_str.lines().skip(1) {
                    if let Ok(mem) = line.trim().parse::<usize>() {
                        return mem;
                    }
                }
            }
        }
    }

    // Fallback: estimate 8GB
    8 * 1024 * 1024 * 1024
}
