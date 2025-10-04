//! Thread-safe callback system using queue+poll pattern
//!
//! This module provides a safe way for Rust code to send messages to C# without
//! the threading issues associated with direct reverse P/Invoke callbacks.

use std::sync::Arc;
use crossbeam_queue::SegQueue;
use std::ffi::CStr;
use std::os::raw::{c_char, c_uchar, c_void};
use std::panic;
use std::sync::atomic::{AtomicPtr, Ordering};

// Global message queue for thread-safe message delivery
lazy_static::lazy_static! {
    static ref MESSAGE_QUEUE: Arc<SegQueue<String>> = Arc::new(SegQueue::new());
}

/// Global callback storage (for legacy direct callback support)
static GLOBAL_CALLBACK: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static GLOBAL_USER_DATA: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// Callback type signature matching the recommended pattern
pub type TextCallback = extern "C" fn(user_data: *mut c_void, data: *const c_uchar, len: usize);

/// Register a direct callback (legacy support - not recommended for multi-threaded use)
#[no_mangle]
pub extern "C" fn register_text_callback(cb: Option<TextCallback>, user_data: *mut c_void) {
    // Store the callback pointer atomically
    let cb_ptr = cb.map(|cb| cb as *const () as *mut c_void).unwrap_or(std::ptr::null_mut());
    GLOBAL_CALLBACK.store(cb_ptr, Ordering::SeqCst);
    GLOBAL_USER_DATA.store(user_data, Ordering::SeqCst);
}

/// Enqueue a message from Rust worker threads
/// This is thread-safe and can be called from any thread
pub fn enqueue_message(message: String) {
    MESSAGE_QUEUE.push(message);
}

/// Enqueue a message from C FFI (for external callers)
#[no_mangle]
pub extern "C" fn enqueue_c_message(msg: *const c_char) {
    if msg.is_null() { return; }

    let res = panic::catch_unwind(|| {
        unsafe {
            if let Ok(s) = CStr::from_ptr(msg).to_str() {
                enqueue_message(s.to_string());
            }
        }
    });

    if res.is_err() {
        // Log error silently - in production you might want to handle this
    }
}

/// Poll for the next message (blocking)
/// Returns the length of the message, 0 if no message available
///
/// # Safety
/// - `buf` must be a valid pointer to at least `buf_len` bytes
/// - Caller must copy the data before calling this function again
#[no_mangle]
pub extern "C" fn poll_next_message(buf: *mut u8, buf_len: usize) -> usize {
    if buf.is_null() || buf_len == 0 {
        return 0;
    }

    if let Some(msg) = MESSAGE_QUEUE.pop() {
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(buf_len);

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, copy_len);
        }

        return copy_len;
    }

    0 // No message available
}

/// Get the length of the next message without removing it from queue
/// Returns 0 if no message available
#[no_mangle]
pub extern "C" fn peek_next_message_length() -> usize {
    // Since SegQueue doesn't support peeking, we'll use a different approach
    // We could try to pop and then push back, but that's not thread-safe
    // For now, return 0 to indicate no message length info available
    // In a production system, you might use a different queue that supports peeking
    0
}

/// Check if there are any messages available
#[no_mangle]
pub extern "C" fn has_messages() -> bool {
    !MESSAGE_QUEUE.is_empty()
}

/// Clear all pending messages (useful for cleanup)
#[no_mangle]
pub extern "C" fn clear_messages() {
    while MESSAGE_QUEUE.pop().is_some() {
        // Drain the queue
    }
}

/// Send a message using the legacy direct callback (not recommended for multi-threaded use)
/// This function is kept for backward compatibility but should be avoided in new code
pub fn send_direct_callback(phase: &str, _current: usize, _total: usize, percent: f32, message: &str) {
    let cb_ptr = GLOBAL_CALLBACK.load(Ordering::SeqCst);
    let user_data = GLOBAL_USER_DATA.load(Ordering::SeqCst);

    if !cb_ptr.is_null() {
        let cb: TextCallback = unsafe { std::mem::transmute(cb_ptr) };

        // Create formatted message
        let full_message = format!("[{}] {} ({:.1}%)", phase, message, percent);
        let message_bytes = full_message.as_bytes();

        // Use catch_unwind to prevent panics from crossing FFI boundary
        let res = panic::catch_unwind(|| {
            cb(user_data, message_bytes.as_ptr(), message_bytes.len());
        });

        if res.is_err() {
            // Log error silently - don't let panic escape
        }
    }
}

/// Macro for thread-safe progress reporting using the queue system
#[macro_export]
macro_rules! thread_safe_progress {
    ($phase:expr, $current:expr, $total:expr, $percent:expr, $message:expr) => {
        let full_message = format!("[{}] {} ({:.1}%)", $phase, $message, $percent);
        $crate::callback_queue::enqueue_message(full_message);
    };
}

/// Macro for legacy direct callback reporting (use with caution)
#[macro_export]
macro_rules! legacy_progress_callback {
    ($phase:expr, $current:expr, $total:expr, $percent:expr, $message:expr) => {
        $crate::callback_queue::send_direct_callback($phase, $current, $total, $percent, $message);
    };
}