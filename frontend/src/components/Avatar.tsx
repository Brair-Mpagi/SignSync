/**
 * Three.js avatar (plan §8.8).
 *
 * Renders the rig as bones in WebGL and drives it from the animation payload.
 * A production build would skin a mesh to this skeleton; the bone rendering is
 * what makes the motion reviewable by a Deaf evaluator before that mesh exists
 * (plan §14 requires every motion iteration to be reviewed before advancing).
 *
 * The camera looks at the signer face-on and the scene is mirrored, because an
 * unmirrored view makes a right-handed sign read as left-handed.
 */

import { useEffect, useRef } from 'react';
import * as THREE from 'three';

import type { AnimationData, Rig } from '../api';
import { boneIndices, forwardKinematics } from '../kinematics';

interface AvatarProps {
  rig: Rig | null;
  animation: AnimationData | null;
  onGlossChange?: (gloss: string | null) => void;
}

export function Avatar({ rig, animation, onGlossChange }: AvatarProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    bones: THREE.LineSegments;
    joints: THREE.Points;
  } | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1526);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, -0.9, 4.2);
    camera.lookAt(0, -0.9, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientWidth * 0.9);
    mount.appendChild(renderer.domElement);

    const bones = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xe2e8f0 }),
    );
    const joints = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ color: 0x60a5fa, size: 0.05 }),
    );
    scene.add(bones, joints);

    stateRef.current = { renderer, scene, camera, bones, joints };
    renderer.render(scene, camera);

    const resize = () => {
      renderer.setSize(mount.clientWidth, mount.clientWidth * 0.9);
      camera.aspect = 1 / 0.9;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state || !rig || !animation || animation.frames.length === 0) return;

    const bonePairs = boneIndices(rig);
    let frameIndex = 0;
    let raf = 0;
    let last = performance.now();
    const frameDuration = 1000 / (animation.fps || 30);
    let currentGloss: string | null = null;

    const positions = new Float32Array(bonePairs.length * 6);
    const jointPositions = new Float32Array(rig.joints.length * 3);
    state.bones.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    state.joints.geometry.setAttribute('position', new THREE.BufferAttribute(jointPositions, 3));

    const draw = () => {
      const world = forwardKinematics(rig, animation.frames[frameIndex]);

      bonePairs.forEach(([parent, child], i) => {
        // Mirror x and flip y: the rig uses image coordinates (y downward) and the
        // viewer faces the signer.
        positions.set(
          [
            -world[parent][0], -world[parent][1], world[parent][2],
            -world[child][0], -world[child][1], world[child][2],
          ],
          i * 6,
        );
      });
      world.forEach((p, i) => jointPositions.set([-p[0], -p[1], p[2]], i * 3));

      state.bones.geometry.attributes.position.needsUpdate = true;
      state.joints.geometry.attributes.position.needsUpdate = true;
      state.renderer.render(state.scene, state.camera);
    };

    const tick = (now: number) => {
      if (now - last >= frameDuration) {
        last = now;
        frameIndex = (frameIndex + 1) % animation.frames.length;

        const time = frameIndex / (animation.fps || 30);
        const segment = animation.segments.find((s) => time >= s.start && time < s.end);
        const gloss = segment?.gloss ?? null;
        if (gloss !== currentGloss) {
          currentGloss = gloss;
          onGlossChange?.(gloss);
        }
        draw();
      }
      raf = requestAnimationFrame(tick);
    };

    draw();
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [rig, animation, onGlossChange]);

  return <div ref={mountRef} className="avatar" />;
}
