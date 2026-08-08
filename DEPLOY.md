# DEPLOY — V.I. STT wrapper (M4)

## picklOS#444 — persistent-engine cutover (com.vi.whisper-engine)

**Status: NOT deployed.** This section is the runbook for the supervised
daytime cutover. Run the infrastructure-change gate
(`Project VI - V0/docs/checklists/infrastructure-change.md`) before AND after.
All commands run on the M4 as `tomtomxyz`.

What ships: `server-wrapper.py` keeps :8178 and its entire contract; when
`WHISPER_ENGINE_URL` is set it POSTs the ffmpeg-normalized WAV to a persistent
`whisper-server` on loopback :8380 instead of spawning `whisper-cli` per
request (S419 spike: 0.72s → ~0.15s warm, 7.4s cold class eliminated). Any
engine failure falls back per-request to the spawn path with one loud
ISO-stamped log line — **the fallback IS the rollback**. Env unset →
byte-identical to pre-#444 behavior.

### Install order

1. **Engine first** (wrapper untouched, zero client risk):

   ```bash
   cp ~/projects/whisper.cpp/com.vi.whisper-engine.plist.draft \
      ~/Library/LaunchAgents/com.vi.whisper-engine.plist
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.vi.whisper-engine.plist
   ```

2. **Verify :8380 with one corpus clip** (corpus lives on M1 at
   `~/projects/whisper.cpp/spike-444-s419/corpus/` — scp one clip over, or
   use any 16kHz mono WAV):

   ```bash
   curl -s http://127.0.0.1:8380/inference -F file=@clip01.wav
   # expect {"text":" ..."} — leading space + trailing \n are the engine's
   # raw contract; the wrapper strips them (contract delta 3)
   ```

3. **Point the wrapper at the engine**: add to
   `~/Library/LaunchAgents/com.vi.whisper-server.plist` (inside `<dict>`):

   ```xml
   <key>EnvironmentVariables</key>
   <dict>
       <key>WHISPER_ENGINE_URL</key>
       <string>http://127.0.0.1:8380/inference</string>
   </dict>
   ```

   then:

   ```bash
   launchctl kickstart -k gui/$(id -u)/com.vi.whisper-server
   ```

### Post-cutover verify

Replay the S419 corpus (M1: `~/projects/whisper.cpp/spike-444-s419/`, bench
script + per-round JSONs) through the real :8178 contract and **byte-diff the
response bodies against the saved pre-cutover captures** in that directory
(`warm_prod*.json`). Also check `~/.local/log/whisper-server.log` for
`ENGINE FALLBACK` lines — a healthy cutover has zero; any line names the
failure class. Nondeterminism caveat: reused whisper_state can flip a
marginal word on near-tie clips (S419 trap 2) — a single flipped word on a
near-tie clip is that known class, not a contract break; the final S419
30-request run had zero flips.

### Rollback

```bash
# 1. Wrapper back to the spawn path (the known-good arm):
#    remove the WHISPER_ENGINE_URL EnvironmentVariables block from
#    com.vi.whisper-server.plist, then
launchctl kickstart -k gui/$(id -u)/com.vi.whisper-server

# 2. Retire the engine:
launchctl bootout gui/$(id -u)/com.vi.whisper-engine
rm ~/Library/LaunchAgents/com.vi.whisper-engine.plist
```

Mid-flight engine death needs no operator action at all: every request
falls back to the spawn path automatically (one loud log line per attempt).

### Gate reminders

- Infra gate surfaces: CLAUDE.md cluster index (unchanged — same cluster),
  `ref_infrastructure.md` (new agent + port 8380), Health Monitor,
  plist backup → `docs/system-config/launchd/`, system-graph regen
  (`generate.py` + `render-html.py`, run LAST).
- Upstream-issue search for the reused-whisper_state nondeterminism is still
  owed (picklOS#444, S419 trap 2) — do it before or at cutover.
