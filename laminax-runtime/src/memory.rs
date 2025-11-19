//! Memory management for heterogeneous computing
//!
//! Handles allocation, deallocation, and data transfer across different memory spaces.

use std::rc::Rc;
use super::{Backend, Device, Result};
use laminax_types::{DType, Shape};
use std::sync::Arc;

/// Abstract buffer handle
#[derive(Clone)]
pub struct Buffer<B: DeviceBuffer> {
    pub id: usize,
    pub shape: Shape,
    pub dtype: DType,
    pub device: Arc<<B::Backend as Backend>::Device>,
    pub data: Arc<B>,
}

#[derive(Default, Clone, Eq, PartialEq, Debug)]
pub struct BufferProperties {
    pub copy_source: Option<bool>,
    pub copy_destination: Option<bool>,
    pub device_local: Option<bool>,
    pub device_write: Option<bool>,
    pub host_visible: Option<bool>,
}

pub trait MemoryManager {
    type Device: Device;

    type Storage: DeviceBuffer;
    type FlushTarget;

    fn allocate(self: Arc<Self>, shape: Shape, dtype: DType, properties: BufferProperties) -> Result<Buffer<Self::Storage>>;

    fn deallocate(self: Arc<Self>, buffer: &Buffer<Self::Storage>) -> Result<()>;

    fn copy(self: Arc<Self>, src: &Buffer<Self::Storage>, dst: &Buffer<Self::Storage>) -> Result<()>;

    fn require_staging(&self) -> bool;
    
    fn stage(&self, data: Vec<u8>, target: &Buffer<Self::Storage>) -> Result<()>;
    
    fn flush(&self, flush_to: &mut Self::FlushTarget) -> Result<()>;
}

pub trait DeviceBuffer: Send + Sync {
    type Backend: Backend;
}

pub trait ShallowClone: Clone {}

impl<T> ShallowClone for Arc<T> {}
impl<T> ShallowClone for Rc<T> {}

impl<B: DeviceBuffer> ShallowClone for Buffer<B> where Buffer<B>: Clone {}
