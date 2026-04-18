$linuxRoot = "/home/frankkkz/qwen35_2b_vla"
$linuxScript = "$linuxRoot/ops/tmux/launch_qwen35_pi05_native_tmux.sh"
$argString = ($args -join " ")

wsl bash -lc "chmod +x $linuxRoot/ops/tmux/*.sh && $linuxScript $argString"
