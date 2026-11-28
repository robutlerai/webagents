//! Networking for a device that holds its own connection to the platform, with
//! no heap: one outbound WebSocket (the device never runs a server), the
//! UAMP frames it sends, a reader for the JSON frames it receives, and a
//! driver for an ESP-AT network co-processor, and (feature `hal`) a driver for
//! the WIZnet W5500 Ethernet controller.
//!
//! Everything works over `embedded-io` byte streams, so the same code runs over
//! a plain TCP socket, a TLS session (`embedded-tls`), or a co-processor link.

pub mod esp_at;
pub mod jsonv;
pub mod uamp;
#[cfg(feature = "hal")]
#[cfg_attr(docsrs, doc(cfg(feature = "hal")))]
pub mod w5500;
pub mod ws;
