import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const html = readFileSync(fileURLToPath(new URL('../../servo_control.html', import.meta.url)), 'utf8');
const start = html.indexOf('    const RESOLUTION = 1024;');
const end = html.indexOf('    const states = new Map(', start);
assert.ok(start >= 0 && end > start);
const logic = new Function(`${html.slice(start, end)}\nreturn { calibration, rawToUnit, unitToRaw, segmentForRaw, Scs215SeamController };`)();

test('all six ranges match the continuous calibration saved on 2026-09-26', () => {
  const expected = new Map([
    [1, [216, 941]],
    [2, [212, 912]],
    [3, [205, 814]],
    [4, [247, 927]],
    [5, [3, 1022]],
    [6, [336, 755]],
  ]);
  for (const config of logic.calibration) {
    assert.deepEqual([config.min, config.max], expected.get(config.id));
    assert.deepEqual(config.segments, [expected.get(config.id)]);
    assert.equal(config.driveMode, 0);
    assert.equal(logic.rawToUnit(config.min, config), 0);
    assert.equal(logic.rawToUnit(config.max, config), 1);
    assert.equal(logic.unitToRaw(0, config), config.min);
    assert.equal(logic.unitToRaw(1, config), config.max);
  }
});

test('continuous mapping is linear throughout the range and clamps without wrapping outside it', () => {
  for (const config of logic.calibration) {
    let previous = -1;
    for (let raw = 0; raw < 1024; raw++) {
      const value = logic.rawToUnit(raw, config);
      const expected = Math.max(0, Math.min(1, (raw - config.min) / (config.max - config.min)));
      assert.ok(Math.abs(value - expected) < 1e-12, `ID ${config.id}, raw ${raw}`);
      assert.ok(value >= previous);
      assert.equal(logic.unitToRaw(value, config), Math.max(config.min, Math.min(config.max, raw)));
      previous = value;
    }
  }
});

// Historical two-segment fixtures exercise the retained generic seam controller;
// they are deliberately independent of the current robot's embedded calibration.
const wrappedConfig = { min: 776, max: 1215, segments: [[776, 1023], [0, 191]] };

function createApp(initialRaw, { id = 1, configOverride = null } = {}) {
  const elements = new Map();
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) {
        elements.set(selector, {
          min: selector === '#speedSlider' ? '1' : '0',
          max: selector === '#speedSlider' ? '1500' : '1',
          value: selector === '#speedSlider' ? '300' : '0.5',
          style: { setProperty() {} },
          classList: { add() {}, remove() {}, toggle() {} },
          addEventListener() {},
          insertAdjacentHTML() {},
        });
      }
      return elements.get(selector);
    },
  };
  const navigator = { serial: { addEventListener() {} } };
  const script = html.slice(html.indexOf('<script>') + 8, html.indexOf('</script>'));
  const app = new Function('document', 'navigator', 'performance', `${script}\nreturn { bus, states, queueTarget, rawToUnit, advanceControl, updatePosition, changeTorque, stopControllers };`)(document, navigator, performance);
  app.bus.port = { readable: true, writable: true };
  const state = app.states.get(id);
  if (configOverride) Object.assign(state.config, configOverride);
  state.online = true;
  state.raw = initialRaw;
  state.actual = app.rawToUnit(initialRaw, state.config);
  state.target = state.actual;
  const memory = new Uint8Array(80);
  const writes = [];
  const setWord = (address, value) => { memory[address] = value >> 8; memory[address + 1] = value & 255; };
  const getWord = address => (memory[address] << 8) | memory[address + 1];
  setWord(11, 1023); setWord(46, 300); setWord(56, initialRaw); memory[48] = 1;
  app.bus.readBytes = async (_id, address, length) => ({ bytes: Array.from(memory.slice(address, address + length)), error: 0 });
  app.bus.writeWord = async (_id, address, value) => {
    if (address === 42 && getWord(11) !== 0) {
      assert.equal(logic.segmentForRaw(value, state.config), logic.segmentForRaw(getWord(56), state.config), 'native position command cannot jump across the seam');
    }
    writes.push([address, value]); setWord(address, value); return { error: 0 };
  };
  app.bus.writeByte = async (_id, address, value) => { writes.push([address, value]); memory[address] = value; return { error: 0 }; };
  const tick = async (raw, now) => {
    setWord(56, raw); app.updatePosition(id, raw, 0); await app.advanceControl(id, raw, now);
  };
  return { app, state, writes, memory, getWord, tick };
}

test('current calibration controls every slider in both directions without PWM transitions', async () => {
  for (const config of logic.calibration) {
    const { app, state, writes, getWord, tick } = createApp(config.min, { id: config.id });
    let raw = config.min;
    for (const [index, unit] of [1, 0, 0.5].entries()) {
      assert.equal(app.queueTarget(config.id, unit), true);
      await tick(raw, index * 0.02);
      const goal = config.min + Math.round(unit * (config.max - config.min));
      assert.equal(getWord(42), goal);
      assert.equal(state.control.phase, 'position');
      assert.equal(state.commandError, null);
      raw = goal;
    }
    assert.equal(getWord(11), 1023);
    assert.ok(writes.every(([address, value]) => address !== 44 || value === 0));
    assert.ok(writes.every(([address]) => address !== 9 && address !== 11));
  }
});

for (const [samples, target, pwm] of [[[1023, 0, 8, 10], 20, 1204], [[0, 1023, 1015, 1013], 1000, 180]]) {
  test(`slider crosses ${samples[0]} to ${target} through directional PWM and settles in position mode`, async () => {
    const { app, state, writes, getWord, tick } = createApp(samples[0], { configOverride: wrappedConfig });
    assert.equal(app.queueTarget(1, app.rawToUnit(target, state.config)), true);
    for (const [index, raw] of samples.entries()) await tick(raw, index * 0.02);
    assert.ok(writes.some(([address, value]) => address === 44 && value === pwm));
    assert.equal(getWord(11), 1023);
    assert.equal(getWord(44), 0);
    assert.equal(getWord(42), target);
    assert.equal(state.control.phase, 'position');
  });
}

test('latest target replaces the approach, and torque-off cancels an ongoing bridge', async () => {
  const { app, state, getWord, memory, tick } = createApp(900, { configOverride: wrappedConfig });
  app.queueTarget(1, app.rawToUnit(20, state.config));
  await tick(900, 0);
  assert.equal(getWord(42), 1011);
  app.queueTarget(1, app.rawToUnit(850, state.config));
  await tick(910, 0.02);
  assert.equal(getWord(42), 850);
  app.queueTarget(1, app.rawToUnit(20, state.config));
  await tick(1010, 0.04);
  assert.equal(getWord(11), 0);
  await app.changeTorque(1, false);
  await tick(1011, 0.06);
  assert.equal(memory[40], 0);
  assert.equal(getWord(44), 0);
  assert.equal(getWord(11), 1023);
  assert.equal(state.controlActive, false);
});

test('missing re-entry feedback stops PWM and restores position mode', async () => {
  const { app, state, getWord, memory, tick } = createApp(1023, { configOverride: wrappedConfig });
  app.queueTarget(1, app.rawToUnit(20, state.config));
  await tick(1023, 0);
  await tick(1023, 3.1);
  assert.equal(getWord(44), 0);
  assert.equal(getWord(11), 1023);
  assert.equal(memory[40], 0);
  assert.ok(state.commandError);
});
