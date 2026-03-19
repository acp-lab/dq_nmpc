#!/bin/bash
echo ""
echo "Let's build the NMPC!"
echo "enter your platform_type"
echo "default: race"
echo 'options: race race2 race_S voxl2 raxl2 iris eagle'
echo ""
read platform_type
platform_type=${platform_type:-race}
echo 'thank you!'
echo ""

python3 dq_nmpc/dq_controller_ros2_node.py $COLCON_PAYLOAD_WS_DIR/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml
