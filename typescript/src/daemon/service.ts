/**
 * Service Installer
 *
 * Generates and installs system service files for the daemon:
 * - macOS: launchd plist
 * - Linux: systemd unit
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { execSync } from 'node:child_process';

export interface ServiceConfig {
  name?: string;
  description?: string;
  port?: number;
  watchDir?: string;
  user?: string;
  nodeExec?: string;
  cliPath?: string;
}

const SERVICE_NAME = 'com.robutler.webagentsd';

export function generateLaunchdPlist(config: ServiceConfig = {}): string {
  const label = config.name ?? SERVICE_NAME;
  const port = config.port ?? 8080;
  const nodeExec = config.nodeExec ?? process.execPath;
  const cliPath = config.cliPath ?? path.resolve(import.meta.dirname ?? __dirname, '../cli/index.js');

  const args = [nodeExec, cliPath, 'daemon', '--port', String(port)];
  if (config.watchDir) args.push('--watch', config.watchDir);

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${label}</string>
  <key>ProgramArguments</key>
  <array>
${args.map((a) => `    <string>${a}</string>`).join('\n')}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${path.join(os.homedir(), '.webagents', 'daemon.log')}</string>
  <key>StandardErrorPath</key>
  <string>${path.join(os.homedir(), '.webagents', 'daemon.err')}</string>
</dict>
</plist>`;
}

export function generateSystemdUnit(config: ServiceConfig = {}): string {
  const description = config.description ?? 'WebAgents Daemon';
  const port = config.port ?? 8080;
  const user = config.user ?? os.userInfo().username;
  const nodeExec = config.nodeExec ?? process.execPath;
  const cliPath = config.cliPath ?? path.resolve(import.meta.dirname ?? __dirname, '../cli/index.js');

  let execStart = `${nodeExec} ${cliPath} daemon --port ${port}`;
  if (config.watchDir) execStart += ` --watch ${config.watchDir}`;

  return `[Unit]
Description=${description}
After=network.target

[Service]
Type=simple
User=${user}
ExecStart=${execStart}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target`;
}

export function installService(config: ServiceConfig = {}): void {
  const platform = os.platform();

  if (platform === 'darwin') {
    const plist = generateLaunchdPlist(config);
    const plistPath = path.join(os.homedir(), 'Library', 'LaunchAgents', `${SERVICE_NAME}.plist`);
    fs.mkdirSync(path.dirname(plistPath), { recursive: true });
    fs.writeFileSync(plistPath, plist);
    console.log(`Wrote launchd plist: ${plistPath}`);
    try {
      execSync(`launchctl load ${plistPath}`);
      console.log('Service loaded. It will start automatically on boot.');
    } catch (err) {
      console.warn('Could not load service:', (err as Error).message);
      console.log(`Manually load with: launchctl load ${plistPath}`);
    }
  } else if (platform === 'linux') {
    const unit = generateSystemdUnit(config);
    const unitPath = `/etc/systemd/system/webagentsd.service`;

    try {
      fs.writeFileSync(unitPath, unit);
      execSync('systemctl daemon-reload');
      execSync('systemctl enable webagentsd');
      console.log(`Wrote systemd unit: ${unitPath}`);
      console.log('Service enabled. Start with: systemctl start webagentsd');
    } catch {
      const localPath = path.join(os.homedir(), '.webagents', 'webagentsd.service');
      fs.mkdirSync(path.dirname(localPath), { recursive: true });
      fs.writeFileSync(localPath, unit);
      console.log(`Wrote systemd unit: ${localPath}`);
      console.log('Copy to /etc/systemd/system/ and run:');
      console.log('  sudo cp ~/.webagents/webagentsd.service /etc/systemd/system/');
      console.log('  sudo systemctl daemon-reload');
      console.log('  sudo systemctl enable --now webagentsd');
    }
  } else {
    console.log('Automatic service install not supported on this platform.');
    console.log('Run the daemon manually: webagents daemon');
  }
}

export function uninstallService(): void {
  const platform = os.platform();

  if (platform === 'darwin') {
    const plistPath = path.join(os.homedir(), 'Library', 'LaunchAgents', `${SERVICE_NAME}.plist`);
    try {
      execSync(`launchctl unload ${plistPath}`);
      fs.unlinkSync(plistPath);
      console.log('Service uninstalled.');
    } catch (err) {
      console.warn('Uninstall error:', (err as Error).message);
    }
  } else if (platform === 'linux') {
    try {
      execSync('systemctl disable --now webagentsd');
      fs.unlinkSync('/etc/systemd/system/webagentsd.service');
      execSync('systemctl daemon-reload');
      console.log('Service uninstalled.');
    } catch (err) {
      console.warn('Uninstall error:', (err as Error).message);
    }
  }
}
