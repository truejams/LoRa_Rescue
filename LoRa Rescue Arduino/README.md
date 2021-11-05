# LoRa Rescue Arduino Code
Build v0.4
- Added support for hop data similar to v0.2
  - From C to mobile node to gateway
  - Address for C is changed to CA

Build v0.3 11-05-21
- Reworked to allow for multiple mobile node transmission
- Reworked addresses, M for Mobile node and A, B, and C for gateway nodes
- Simplified most of the code
- Removed hop support

Build v0.2 10-27-21
- Changed system to hop data from 'C -> mobile node -> gateway' to preserve LOS environment during testing and avoid data loss
  - This is to compensate for testing on NLOS environments
- Adjusted system to accept character arrays instead of strings to avoid overloading Arduino memory
- Adjusted system to always send address bytes
- Delayed C data sending to avoid signal overlapping in receivers
  - This delays until B is finished sending which is approximately 9 seconds based on indoor setting
  - The delay was adjusted to 12 seconds to compensate for system lag
- Possible undocumented changes
- Changes only apply to Mobile Node, Main Gateway, Gateway B, and Gateway C


Build v0.1 07-09-21
- Original build used in prior testing
- Updated to send RSSI and not distance