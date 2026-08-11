/**
 * Skeleton renderer for the SignSync avatar.
 *
 * Deliberately dependency-free 2D canvas. Plan §17 expects deployments with poor
 * or no connectivity, so the client that ships with the server cannot pull a 3D
 * library from a CDN — it has to work from a laptop that has never been online.
 * The React client under frontend/ renders the same animation format through
 * Three.js when a build step is available.
 *
 * The wire format is joint offsets plus per-frame quaternions, so forward
 * kinematics happens here (see plan §8.8 and signsync/avatar/export.py).
 */

/** Rotate a vector by a quaternion (w, x, y, z). */
function rotate(q, v) {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  // t = 2 * (q_vec x v); v' = v + w*t + q_vec x t
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

function multiply(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

/**
 * World-space joint positions for one frame.
 * Parents always precede children in the rig, which the server guarantees.
 */
export function forwardKinematics(rig, frame) {
  const positions = new Array(rig.joints.length);
  const worldRotations = new Array(rig.joints.length);
  const index = new Map(rig.joints.map((joint, i) => [joint.name, i]));

  rig.joints.forEach((joint, i) => {
    const local = frame.rotations[i] || [1, 0, 0, 0];
    if (joint.parent === null || joint.parent === undefined) {
      worldRotations[i] = local;
      positions[i] = [
        (frame.root?.[0] ?? 0) + joint.offset[0],
        (frame.root?.[1] ?? 0) + joint.offset[1],
        (frame.root?.[2] ?? 0) + joint.offset[2],
      ];
    } else {
      const p = index.get(joint.parent);
      worldRotations[i] = multiply(worldRotations[p], local);
      const offset = rotate(worldRotations[p], joint.offset);
      positions[i] = [
        positions[p][0] + offset[0],
        positions[p][1] + offset[1],
        positions[p][2] + offset[2],
      ];
    }
  });
  return positions;
}

const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];

export class AvatarRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.rig = null;
    this.animation = null;
    this.frameIndex = 0;
    this.playing = false;
    this.onFrame = null;
    this._lastTime = 0;
  }

  setRig(rig) {
    this.rig = rig;
    this._bones = rig.joints
      .map((joint, i) => [rig.joints.findIndex((j) => j.name === joint.parent), i])
      .filter(([parent]) => parent >= 0);
  }

  setAnimation(animation) {
    this.animation = animation;
    this.frameIndex = 0;
  }

  play() {
    if (!this.animation || this.playing) return;
    this.playing = true;
    this._lastTime = performance.now();
    requestAnimationFrame((t) => this._tick(t));
  }

  stop() {
    this.playing = false;
  }

  _tick(now) {
    if (!this.playing || !this.animation) return;
    const fps = this.animation.fps || 30;
    const elapsed = (now - this._lastTime) / 1000;
    if (elapsed >= 1 / fps) {
      this._lastTime = now;
      this.frameIndex += 1;
      if (this.frameIndex >= this.animation.frames.length) {
        this.frameIndex = this.animation.frames.length - 1;
        this.playing = false;
        this.draw();
        if (this.onFrame) this.onFrame(this.frameIndex, null);
        return;
      }
      this.draw();
    }
    requestAnimationFrame((t) => this._tick(t));
  }

  /** Current gloss, so the UI can show what is being signed right now. */
  currentGloss() {
    if (!this.animation) return null;
    const time = this.frameIndex / (this.animation.fps || 30);
    const segment = (this.animation.segments || []).find((s) => time >= s.start && time < s.end);
    return segment ? segment.gloss : null;
  }

  draw() {
    const { context: ctx, canvas } = this;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (!this.rig || !this.animation || this.animation.frames.length === 0) {
      ctx.fillStyle = 'rgba(148,163,184,0.9)';
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No animation loaded', width / 2, height / 2);
      return;
    }

    const frame = this.animation.frames[Math.min(this.frameIndex, this.animation.frames.length - 1)];
    const positions = forwardKinematics(this.rig, frame);

    // Orthographic projection. The rig's +y is downward (image convention), and
    // one unit is one shoulder width, so the scale is chosen from the canvas.
    const scale = Math.min(width, height) * 0.34;
    const originX = width / 2;
    const originY = height * 0.16;
    const project = ([x, y, z]) => [originX - x * scale, originY + y * scale, z];

    // Mirrored horizontally: the viewer is facing the signer, and an unmirrored
    // view makes a right-handed sign look left-handed.
    const points = positions.map(project);

    const depths = points.map((p) => p[2]);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);
    const depthRange = Math.max(maxDepth - minDepth, 1e-3);

    ctx.lineCap = 'round';
    for (const [parent, child] of this._bones) {
      const name = this.rig.joints[child].name;
      const isFinger = FINGERS.some((f) => name.includes(f));
      const depth = (points[child][2] - minDepth) / depthRange;

      ctx.strokeStyle = isFinger
        ? `rgba(96,165,250,${0.45 + 0.5 * depth})`
        : `rgba(226,232,240,${0.5 + 0.5 * depth})`;
      ctx.lineWidth = isFinger ? 2.2 : 6.5;
      ctx.beginPath();
      ctx.moveTo(points[parent][0], points[parent][1]);
      ctx.lineTo(points[child][0], points[child][1]);
      ctx.stroke();
    }

    this._drawHead(points, scale, frame.face || {});
    this._drawWrists(points, scale);
  }

  _drawHead(points, scale, face) {
    const ctx = this.context;
    const headIndex = this.rig.joints.findIndex((j) => j.name === 'head');
    if (headIndex < 0) return;
    const [x, y] = points[headIndex];
    const radius = scale * 0.22;

    ctx.fillStyle = 'rgba(226,232,240,0.92)';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Non-manual markers are grammar, so they are drawn, not implied: brow height
    // for question type, a shake indicator for negation (plan §8.7).
    const raise = face.brow_raise || 0;
    const furrow = face.brow_furrow || 0;
    const browY = y - radius * (0.36 + 0.18 * raise - 0.05 * furrow);

    ctx.strokeStyle = furrow > 0.1 ? 'rgba(248,113,113,0.95)' : 'rgba(30,41,59,0.9)';
    ctx.lineWidth = Math.max(2, radius * 0.12);
    for (const side of [-1, 1]) {
      ctx.beginPath();
      const inner = x + side * radius * 0.12;
      const outer = x + side * radius * 0.52;
      ctx.moveTo(inner, browY + furrow * radius * 0.12);
      ctx.lineTo(outer, browY - raise * radius * 0.06);
      ctx.stroke();
    }

    ctx.fillStyle = 'rgba(30,41,59,0.9)';
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.arc(x + side * radius * 0.3, y - radius * 0.1, radius * 0.09, 0, Math.PI * 2);
      ctx.fill();
    }

    const open = face.mouth_open || 0;
    ctx.beginPath();
    ctx.ellipse(x, y + radius * 0.42, radius * 0.26, radius * (0.06 + 0.16 * open), 0, 0, Math.PI * 2);
    ctx.fill();

    const shake = face.head_shake || 0;
    if (shake > 0.05) {
      ctx.strokeStyle = `rgba(248,113,113,${Math.min(1, shake)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius * 1.35, Math.PI * 0.85, Math.PI * 0.15, true);
      ctx.stroke();
    }
  }

  _drawWrists(points, scale) {
    const ctx = this.context;
    for (const side of ['left', 'right']) {
      const index = this.rig.joints.findIndex((j) => j.name === `${side}_wrist`);
      if (index < 0) continue;
      ctx.fillStyle = 'rgba(59,130,246,0.85)';
      ctx.beginPath();
      ctx.arc(points[index][0], points[index][1], scale * 0.035, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
