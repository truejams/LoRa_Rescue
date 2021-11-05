# LoRa Rescue Arduino Code
Build v0.1 (This version can be found in the backup folder)
- Original build used in prior testing
- Updated to send RSSI and not distance

Build v0.2
- Changed system to hop data from 'C -> mobile node -> gateway' to preserve LOS environment during testing and avoid data loss
  - This is to compensate for testing on NLOS environments
- Adjusted system to accept character arrays instead of strings to avoid overloading Arduino memory
- Adjusted system to always send address bytes
- Delayed C data sending to avoid signal overlapping in receivers
  - This delays until B is finished sending which is approximately 9 seconds based on indoor setting
  - The delay was adjusted to 12 seconds to compensate for system lag
- Possible undocumented changes
- Changes only apply to Mobile Node, Main Gateway, Gateway B, and Gateway C
