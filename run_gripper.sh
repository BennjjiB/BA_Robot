# Execute the auto_gripper.sh script with the same configuration files
# Define the target directory and configuration files
target_dir="/home/panda3/ws/3rd_party_lib/deoxys_control/deoxys/auto_scripts"
config_file1="../config/charmander.yml"
config_file2="../config/control_config.yml"

# Change to the target directory
cd "$target_dir" || { echo "Failed to change directory to $target_dir"; exit 1; }

# Execute the auto_gripper.sh script with the same configuration files
./auto_gripper.sh "$config_file1" "$config_file2"
