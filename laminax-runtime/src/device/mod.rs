//! Device abstraction layer for heterogeneous computing.
//!
//! Provides unified interfaces for different compute devices (CPU, GPU, etc.)
//! across all Laminax crates.

use crate::cpu::CpuBackend;
use crate::{MemoryManager, RuntimeError, SupportedRuntime};
use std::sync::Arc;

pub mod cpu;

#[cfg(feature = "vulkan")]
pub mod vulkan;
mod metal;
mod cuda;
mod tpu;
mod coral;
mod rocm;

/// Types of compute devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Hip,
    Metal,
    Vulkan,
    Rocm,
    Tpu,
    Coral,
    Custom,
}

#[derive(Debug, Copy, Clone)]
pub struct DeviceCapabilities {
    pub compute_units: usize,
    pub max_work_group_size: usize,
    pub local_memory_size: usize,
    pub global_memory_size: usize,
    pub supports_fp64: bool,
    pub supports_fp16: bool,
    pub supports_async: bool,
    pub unified_memory: bool,
    pub shared_memory: bool,
}

pub trait Backend {
    type Device: Device;
    const BACKEND_TYPE: DeviceType;

    fn discover_devices(&self, required_capabilities: DeviceCapabilities) -> Vec<Arc<Self::Device>>;
}

/// Abstract device interface
pub trait Device: Send + Sync + MemoryManager {
    type Backend: Backend;

    /// Get device type
    fn device_type() -> DeviceType;

    /// Get device capabilities
    fn capabilities(&self) -> DeviceCapabilities;

    /// Check if device is available for use
    fn is_available(&self) -> bool;

    /// Get device name
    fn name(&self) -> &str;
}

impl DeviceCapabilities {
    pub fn satisfies(&self, requirement: &Self) -> bool {
        if self.compute_units < requirement.compute_units {
            return false;
        }
        if self.max_work_group_size < requirement.max_work_group_size {
            return false;
        }
        if self.local_memory_size < requirement.local_memory_size {
            return false;
        }
        if self.global_memory_size < requirement.global_memory_size {
            return false;
        }
        if requirement.supports_fp64 && !self.supports_fp64 {
            return false;
        }
        if requirement.supports_fp16 && !self.supports_fp16 {
            return false;
        }
        if requirement.supports_async && !self.supports_async {
            return false;
        }
        if requirement.unified_memory && !self.unified_memory {
            return false;
        }
        if requirement.shared_memory && !self.shared_memory {
            return false;
        }
        true
    }
}

/// Enumerate all available devices
pub fn enumerate_devices(requirements: DeviceCapabilities) -> Result<Vec<SupportedRuntime>, RuntimeError> {
    let mut devices = Vec::new();

    // Add CPU device
    devices.extend(CpuBackend.discover_devices(requirements).into_iter().map(Into::into));

    #[cfg(feature = "vulkan")]
    {
        use crate::vulkan::VulkanBackend;
        let vulkan_backend = VulkanBackend::new();
        devices.extend(vulkan_backend.discover_devices(requirements).into_iter().map(Into::into))
    }

    Ok(devices)
}
