import os
import sys
import fileinput
import json # 导入json库用于处理agnos.json
import urllib.request # 导入urllib库用于下载文件
import urllib.error   # 导入urllib库用于处理错误

# --- 文件路径 ---
repo_root = os.environ.get('GITHUB_WORKSPACE', '.')  # 默认为当前目录
registration_file = os.path.join(repo_root, "system/athena/registration.py")
launch_script = os.path.join(repo_root, "launch_openpilot.sh")
process_config = os.path.join(repo_root, "system/manager/process_config.py")
long_mpc = os.path.join(repo_root, "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py")
pandad_py = os.path.join(repo_root, "selfdrive/pandad/pandad.py")
pandad_cc = os.path.join(repo_root, "selfdrive/pandad/pandad.cc")
hardwared_py = os.path.join(repo_root, "system/hardware/hardwared.py")
hardware_h = os.path.join(repo_root, "system/hardware/tici/hardware.h")
selfdrived_py = os.path.join(repo_root, "selfdrive/selfdrived/selfdrived.py")
updated_py = os.path.join(repo_root, "system/updated/updated.py")
agnos_json = os.path.join(repo_root, "system/hardware/tici/agnos.json")
software_panel_cc = os.path.join(repo_root, "selfdrive/ui/sunnypilot/qt/offroad/settings/software_panel.cc")
ui_cc = os.path.join(repo_root, "selfdrive/ui/ui.cc")
alerts_h = os.path.join(repo_root, "selfdrive/ui/qt/onroad/alerts.h") # 新增 alerts.h 路径
lfs_config = os.path.join(repo_root, ".lfsconfig")
gitmodules_file = os.path.join(repo_root, ".gitmodules")

# --- Helper for status printing ---
def print_status(filename, modified, message_if_modified, message_if_not_modified="already in desired state"):
    # 如果文件名是URL，则只显示基础名称
    display_name = os.path.basename(filename) if not filename.startswith("http") else filename.split('/')[-1]

    if modified:
        print(f"  {message_if_modified}")
    else:
        print(f"  {display_name} {message_if_not_modified}.")


# --- 下载函数 ---
def download_lfsconfig(filename):
    """
    从指定URL下载.lfsconfig文件并保存到仓库根目录.
    """
    url = "https://github.com/sunnypilot/sunnypilot/raw/refs/heads/tn/.lfsconfig"
    print(f"Downloading {os.path.basename(filename)} from {url}...")

    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                print(f"  Error: Failed to download file. Status code: {response.status}", file=sys.stderr)
                return False
            content = response.read()

        with open(filename, 'wb') as f:
            f.write(content)

        print_status(url, True, f"Successfully downloaded and saved to {os.path.basename(filename)}.")
        return True

    except urllib.error.URLError as e:
        print(f"  Error downloading file: {e.reason}", file=sys.stderr)
        return False
    except IOError as e:
        print(f"  Error writing to file {filename}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  An unexpected error occurred during download: {e}", file=sys.stderr)
        return False


# --- 修改函数 (fileinput 适用于简单的单行替换) ---

def modify_registration(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        stripped_line = line.strip()
        if stripped_line == "imei1: str | None = None":
            print(line.replace(stripped_line, "imei1='865420071781912'"), end='')
            modified = True
        elif stripped_line == "imei2: str | None = None":
            print(line.replace(stripped_line, "imei2='865420071781904'"), end='')
            modified = True
        elif 'set_offroad_alert("Offroad_UnofficialHardware"' in line and not stripped_line.startswith("#"):
            print("#" + line, end='')
            modified = True
        else:
            print(line, end='')
    print_status(filename, modified, "IMEI and/or alert modified.")
    return True

def modify_process_config(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        if ('PythonProcess("dmonitoringmodeld"' in line or 'PythonProcess("dmonitoringd"' in line) and not line.strip().startswith("#"):
            print("#" + line, end='')
            modified = True
        else:
            print(line, end='')
    print_status(filename, modified, "dmonitoring processes commented.")
    return True

def modify_long_mpc(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        if 'STOP_DISTANCE' in line and '=' in line and not line.strip().startswith("#"):
            if line.strip() != "STOP_DISTANCE = 4.5":
                print(line.split('=')[0] + "= 4.5\n", end='')
                modified = True
            else:
                print(line, end='')
        else:
            print(line, end='')
    print_status(filename, modified, "STOP_DISTANCE changed to 4.5.")
    return True

def modify_pandad_py(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        if 'if time.monotonic() < 60.:' in line and 'if time.monotonic() < 45.:' not in line:
            print(line.replace('60.', '40.'), end='')
            modified = True
        else:
            print(line, end='')
    print_status(filename, modified, "time.monotonic limit changed from 35 to 45.")
    return True

def modify_pandad_cc(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        if line.strip() == '#define MAX_IR_PANDA_VAL 50':
            print("#define MAX_IR_PANDA_VAL 0\n", end='')
            modified = True
        else:
            print(line, end='')
    print_status(filename, modified, "MAX_IR_PANDA_VAL changed to 0.")
    return True

def modify_gitmodules(filename):
    """
    为 .gitmodules 文件中的 GitHub URL 添加代理前缀。
    """
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False

    modified = False
    old_url_prefix = "https://github.com/"
    proxy_prefix = "https://gh-proxy.com/"
    new_url_prefix = proxy_prefix + old_url_prefix

    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        if old_url_prefix in line and proxy_prefix not in line:
            print(line.replace(old_url_prefix, new_url_prefix), end='')
            modified = True
        else:
            print(line, end='')

    print_status(filename, modified, "Submodule URLs prefixed with gh-proxy.com.")
    return True

def modify_ui_cc(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    original_line = 'timeout = customTimeout == 0 ? (ignition_on ? 10 : 30) : customTimeout;'
    new_line = 'timeout = customTimeout == 0 ? (ignition_on ? 120 : 360) : customTimeout;'

    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        if original_line in line:
            print(line.replace(original_line, new_line), end='')
            modified = True
        else:
            print(line, end='')

    print_status(filename, modified, "Screen timeout values increased.")
    return True

def modify_alerts_h(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    modified = False
    for line in fileinput.input(filename, inplace=True, encoding="utf-8"):
        original_line = line
        if "{cereal::SelfdriveState::AlertStatus::NORMAL, QColor(0x15, 0x15, 0x15, 0xf1)}" in line:
            line = line.replace("0xf1", "0x80")
        elif "{cereal::SelfdriveState::AlertStatus::USER_PROMPT, QColor(0xDA, 0x6F, 0x25, 0xf1)}" in line:
            line = line.replace("0xf1", "0x99")
        elif "{cereal::SelfdriveState::AlertStatus::CRITICAL, QColor(0xC9, 0x22, 0x31, 0xf1)}" in line:
            line = line.replace("0xf1", "0x99")

        if original_line != line:
            modified = True

        print(line, end='')

    print_status(filename, modified, "Alert color alpha values modified.")
    return True

# --- 修改函数 (read/write 适用于多行、复杂或上下文相关的修改) ---

def modify_hardwared_py(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            # 修改点 1: 修改 import 语句
            #if stripped_line == "from openpilot.system.version import terms_version, training_version, get_build_metadata":
             #   if ", get_build_metadata" in line:
              #      new_lines.append(line.replace(", get_build_metadata", ""))
              #      modified = True
              #  else:
             #       new_lines.append(line)
              #  i += 1
             #   continue

            # 修改点 2: 删除不支持的设备组合检测块
            #if stripped_line == "# if an unsupported device and branch is detected, going onroad is blocked":
            #    if (i + 8 < len(lines) and
             #       'set_offroad_alert("Offroad_TiciSupport"' in lines[i+8]):
              #      i += 9 # 跳过这 9 行
              #      modified = True
               #     continue

            # 修改点 3: 修改存储缺失警报块
            if stripped_line == 'if TICI and HARDWARE.get_device_type() == "tici":':
                 if (i + 3 < len(lines) and
                    'set_offroad_alert_if_changed("Offroad_StorageMissing", True)' in lines[i+3]):

                    indent1 = line[:len(line) - len(line.lstrip())]
                    indent4 = lines[i+3][:len(lines[i+3]) - len(lines[i+3].lstrip())]

                    new_lines.append(f"{indent1}if False:\n")
                    new_lines.append(lines[i+1])
                    new_lines.append(lines[i+2])
                    new_lines.append(f'{indent4}set_offroad_alert_if_changed("Offroad_StorageMissing", False)\n')
                    modified = True
                    i += 4
                    continue

            new_lines.append(line)
            i += 1

        if modified:
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        print_status(文件名, 已修改,“导入、tici 支持块和存储警报已修改。”)
        返回 True

异常 作为是：
        print(f" 错误修改{filename}: {e}", file=sys.stderr)
        返回 False

def modify_launch_script(filename):
    print(f"修改{filename}...")
    if not os.path.exists(filename):
        print(f" 文件未找到：{filename}", file=sys.stderr)
        返回 False
    尝试:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        lines_to_insert = [
            "export API_HOST=https://api.konik.ai\n",
            "export ATHENA_HOST=wss://athena.konik.ai\n",
            "#export MAPS_HOST=https://api.konik.ai/maps\n",
            "export MAPBOX_TOKEN='pk.eyJ1IjoibXJvbmVjYyIsImEiOiJjbHhqbzlkbTYxNXUwMmtzZjdoMGtrZnVvIn0.SC7GNLtMFUGDgC2bAZcKzg'\n"
        ]

        if all(l in lines for l in lines_to_insert):
            print_status(filename, False, "")
            返回 True

        content_without_inserts = [l for l in lines if l not in lines_to_insert]
idx =1如果content_without_inserts且[0].startswith("#!")否则
        new_content = content_without_inserts[:idx] + lines_to_insert + content_without_inserts[idx:]

        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(new_content)

        print_status(filename, True, "Environment lines inserted/updated.")
        返回 True
    except Exception as e:
        print(f"  Error modifying {filename}: {e}", file=sys.stderr)
        return False

def modify_selfdrived_py(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            if stripped_line == "if SIMULATION:":
                if i + 1 < len(lines) and "ignore += ['driverCameraState', 'managerState']" in lines[i+1]:
                    indent = line[:len(line) - len(line.lstrip())]
                    next_indent = lines[i+1][:len(lines[i+1]) - len(lines[i+1].lstrip())]

                    new_lines.append(f"{indent}if True:\n")
                    new_lines.append(f"{next_indent}ignore += ['driverCameraState', 'managerState', 'driverMonitoringState']\n")

                    modified = True
                    i += 2
                    continue

            if stripped_line == "if True:":
                if i + 1 < len(lines) and "ignore += ['driverCameraState', 'managerState', 'driverMonitoringState']" in lines[i+1]:
                    new_lines.append(line)
                    new_lines.append(lines[i+1])
                    i += 2
                    continue

            lines_to_comment = [
                "self.events.add(EventName.commIssue)",
                "self.events.add(EventName.commIssueAvgFreq)",
                "self.events.add(EventName.cameraMalfunction)",
                'cloudlog.event("process_not_running", not_running=not_running, error=True)',
                'self.events.add(EventName.processNotRunning)',
                'self.events.add(EventName.sensorDataInvalid)',
                'self.events.add(EventName.noGps)',
            ]

            is_line_to_comment = stripped_line in lines_to_comment

            if is_line_to_comment and not line.lstrip().startswith(("pass", "#")):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}pass  # {stripped_line}\n")
                modified = True
            else:
                new_lines.append(line)

            i += 1

        if modified:
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        print_status(filename, modified, "SIMULATION block and specified alerts modified.")
        return True
    except Exception as e:
        print(f"  Error modifying {filename}: {e}", file=sys.stderr)
        return False

def modify_updated_py(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            if stripped_line == 'elif failed_count > 0:':
                if i > 0 and not lines[i-1].strip() == "# 关闭长时间不联网限制":
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}# 关闭长时间不联网限制\n")
                    for j in range(6):
                        if i + j < len(lines):
                            block_line = lines[i+j]
                            block_indent = block_line[:len(block_line) - len(block_line.lstrip())]
                            new_lines.append(f"{block_indent}# {block_line.lstrip()}")
                    modified = True
                    i += 6
                    continue

            new_lines.append(line)
            i += 1

        if modified:
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        print_status(filename, modified, "Connectivity limit block commented out.")
        return True
    except Exception as e:
        print(f"  Error modifying {filename}: {e}", file=sys.stderr)
        return False

def modify_hardware_h(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        i = 0
        while i < len(lines):
            line = lines[i]

            if "static void set_ir_power(int percent) {" in line:
                new_lines.append(line)
                if not (i + 1 < len(lines) and "(void)percent;" in lines[i+1]):
                    indent = "    " if not (i + 1 < len(lines) and lines[i+1].startswith(" ")) else lines[i+1][:len(lines[i+1]) - len(lines[i+1].lstrip())]
                    new_lines.append(f"{indent}(void)percent; // 忽略传入参数，避免编译器警告\n")
                    modified = True
                i += 1
                continue

            elif "int value = util::map_val" in line:
                if not (len(new_lines) > 0 and "// 强制设为 0" in new_lines[-1]):
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}// 强制设为 0\n")
                    new_lines.append(f'{indent}std::ofstream("/sys/class/leds/led:switch_2/brightness") << 0 << "\\n";\n')
                    new_lines.append(f'{indent}std::ofstream("/sys/class/leds/led:torch_2/brightness") << 0 << "\\n";\n')
                    new_lines.append(f'{indent}std::ofstream("/sys/class/leds/led:switch_2/brightness") << 0 << "\\n";\n')
                    modified = True
                i += 4
                continue

            new_lines.append(line)
            i += 1

        if modified:
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        print_status(filename, modified, "IR power logic has been modified.")
        return True

    except Exception as e:
        print(f"  Error modifying {filename}: {e}", file=sys.stderr)
        return False

def modify_agnos_json(filename):
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False

    modified = False
    old_prefix = "https://commadist.azureedge.net/agnosupdate/"
    new_prefix = "https://commadist.azureedge.net/agnosupdate/"

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            agnos_data = json.load(f)

        for item in agnos_data:
            if "url" in item and item["url"].startswith(old_prefix):
                item["url"] = item["url"].replace(old_prefix, new_prefix, 1)
                modified = True

            if "alt" in item and isinstance(item["alt"], dict) and "url" in item["alt"]:
                if item["alt"]["url"].startswith(old_prefix):
                    item["alt"]["url"] = item["alt"]["url"].replace(old_prefix, new_prefix, 1)
                    modified = True

        if modified:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(agnos_data, f, indent=2, ensure_ascii=False)
            print_status(filename, modified, "Agnos download URLs updated.")
        else:
            print_status(filename, modified, "")

        return True
    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON in {filename}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Error modifying {filename}: {e}", file=sys.stderr)
        return False

def modify_software_panel_cc(filename):
    """
    修改 software_panel.cc 以简化分支选择逻辑。
    """
    print(f"Modifying {filename}...")
    if not os.path.exists(filename):
        print(f"  File not found: {filename}", file=sys.stderr)
        return False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        content_str = "".join(lines)
        if 'Hardware::get_device_type() == cereal::InitData::DeviceType::TICI' not in content_str:
             print_status(filename, False, "already in desired state")
             return True

        new_lines = []
        modified = False
        in_block_to_replace = False
        block_replaced = False

        for line in lines:
            stripped_line = line.strip()

            if not block_replaced and stripped_line == 'connect(targetBranchBtn, &ButtonControlSP::clicked, [=]() {':
                in_block_to_replace = True
                new_lines.append(line)

                indent1 = "    "
                indent2 = "      "
                new_lines.append(f'{indent1}InputDialog d(tr("Search Branch"), this, tr("Enter search keywords, or leave blank to list all branches."), false);\n')
                new_lines.append(f'{indent2}d.setMinLength(0);\n')
                new_lines.append(f'{indent2}const int ret = d.exec();\n')
                new_lines.append(f'{indent2}if (ret) {{\n')
                new_lines.append(f'{indent2}  searchBranches(d.text());\n')
                new_lines.append(f'{indent2}}}\n')

                modified = True
                block_replaced = True
                continue

            if in_block_to_replace and stripped_line == '});':
                in_block_to_replace = False

            if in_block_to_replace:
                continue

            new_lines.append(line)

        if modified:
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        print_status(filename, modified, "Branch selector logic simplified.")
        return True

    except Exception as e:
        print(f"  Error modifying {filename}: {e}", file=sys.stderr)
        return False

# --- 主入口 ---
if __name__ == "__main__":
    print("Running all modifications...")

    modifications = {
        #"lfs_config": (download_lfsconfig, lfs_config),
        #"gitmodules": (modify_gitmodules, gitmodules_file),
        "registration": (modify_registration, registration_file),
        "launch_script": (modify_launch_script, launch_script),
        "process_config": (modify_process_config, process_config),
        "long_mpc": (modify_long_mpc, long_mpc),
        "pandad_py": (modify_pandad_py, pandad_py),
        "pandad_cc": (modify_pandad_cc, pandad_cc),
        "hardwared_py": (modify_hardwared_py, hardwared_py),
        "selfdrived": (modify_selfdrived_py, selfdrived_py),
        "updated": (modify_updated_py, updated_py),
        #"hardware_h": (modify_hardware_h, hardware_h),
        #"software_panel_cc": (modify_software_panel_cc, software_panel_cc),
        #"ui_cc": (modify_ui_cc, ui_cc),
        #"alerts_h": (modify_alerts_h, alerts_h), # 新增修改项
        #"agnos_json": (modify_agnos_json, agnos_json),
    }

    results = {}
    for name, (func, path) in modifications.items():
        results[name] = func(path)

    if all(results.values()):
        print("\n✅ All modifications applied successfully or files were already in the desired state.")
        sys.exit(0)
    else:
        print("\n❌ Some modifications may have failed or were not applicable.", file=sys.stderr)
        failed_mods = [name for name, success in results.items() if not success]
        if failed_mods:
            print(f"  Potentially failed/unapplied modifications for: {', '.join(failed_mods)}", file=sys.stderr)
        sys.exit(1)
