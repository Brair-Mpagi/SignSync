/**
 * Forward kinematics for the SignSync rig.
 *
 * Must stay numerically identical to `signsync/avatar/rig.py`. The wire format is
 * joint offsets plus per-frame quaternions rather than baked positions, so the
 * client does this maths itself, and a divergence renders a subtly wrong avatar —
 * an arm bent the wrong way is a different sign, not a cosmetic defect.
 *
 * `tests/test_client.py` cross-checks the equivalent function in the shipped
 * dependency-free client against Python. If you change either implementation,
 * change both and re-run that test.
 */

import type { Frame, Rig } from './api';

export type Quaternion = [number, number, number, number]; // w, x, y, z
export type Vector3 = [number, number, number];

export function rotateVector(q: Quaternion, v: Vector3): Vector3 {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

export function multiplyQuaternions(a: Quaternion, b: Quaternion): Quaternion {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

/** World-space joint positions for one frame. Parents always precede children. */
export function forwardKinematics(rig: Rig, frame: Frame): Vector3[] {
  const positions: Vector3[] = new Array(rig.joints.length);
  const world: Quaternion[] = new Array(rig.joints.length);
  const index = new Map(rig.joints.map((joint, i) => [joint.name, i]));

  rig.joints.forEach((joint, i) => {
    const local = (frame.rotations[i] ?? [1, 0, 0, 0]) as Quaternion;
    if (joint.parent === null) {
      world[i] = local;
      const root = frame.root ?? [0, 0, 0];
      positions[i] = [
        root[0] + joint.offset[0],
        root[1] + joint.offset[1],
        root[2] + joint.offset[2],
      ];
    } else {
      const p = index.get(joint.parent)!;
      world[i] = multiplyQuaternions(world[p], local);
      const offset = rotateVector(world[p], joint.offset);
      positions[i] = [
        positions[p][0] + offset[0],
        positions[p][1] + offset[1],
        positions[p][2] + offset[2],
      ];
    }
  });

  return positions;
}

/** Bone pairs as [parentIndex, childIndex], for drawing the skeleton. */
export function boneIndices(rig: Rig): [number, number][] {
  const index = new Map(rig.joints.map((joint, i) => [joint.name, i]));
  return rig.joints
    .map((joint, i): [number, number] => [joint.parent ? (index.get(joint.parent) ?? -1) : -1, i])
    .filter(([parent]) => parent >= 0);
}
