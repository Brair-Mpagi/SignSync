/**
 * Three.js avatar (plan §8.8).
 *
 * Renders the rig as a realistic humanoid mesh in WebGL and drives it from the
 * animation payload. The mesh features detailed hands with clearly visible fingers,
 * natural body proportions, and facial features for sign language communication.
 *
 * The camera looks at the signer face-on and the scene is mirrored, because an
 * unmirrored view makes a right-handed sign read as left-handed.
 *
 * The original skeleton line rendering is available as a debug overlay.
 */

import { useEffect, useRef } from 'react';
import * as THREE from 'three';

import type { AnimationData, Rig } from '../api';
import { boneIndices, forwardKinematics } from '../kinematics';
import { 
  createHumanoidMesh, 
  updateHumanoidMeshFromSkeleton, 
  applyFacialExpression,
  type HumanoidParts 
} from './HumanoidMesh';

interface AvatarProps {
  rig: Rig | null;
  animation: AnimationData | null;
  onGlossChange?: (gloss: string | null) => void;
  showSkeleton?: boolean; // Debug mode: show skeleton overlay
}

export function Avatar({ rig, animation, onGlossChange, showSkeleton = false }: AvatarProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    bones: THREE.LineSegments;
    joints: THREE.Points;
    humanoid: HumanoidParts | null;
    lights: THREE.Light[];
  } | null>(null);
  const humanoidCreated = useRef(false);

  // Initialize the 3D scene
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a2332);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, -0.9, 4.2);
    camera.lookAt(0, -0.9, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientWidth * 0.9);
    mount.appendChild(renderer.domElement);

    // Add realistic lighting for the humanoid mesh
    const lights: THREE.Light[] = [];
    
    // Key light (main light from front-right)
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(2, 2, 4);
    lights.push(keyLight);
    scene.add(keyLight);
    
    // Fill light (softer light from front-left)
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-2, 1, 3);
    lights.push(fillLight);
    scene.add(fillLight);
    
    // Back light (rim light for depth)
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(0, 0, -2);
    lights.push(backLight);
    scene.add(backLight);
    
    // Ambient light (soft overall illumination)
    const ambientLight = new THREE.AmbientLight(0x8899aa, 0.5);
    lights.push(ambientLight);
    scene.add(ambientLight);

    // Skeleton rendering (for debug mode)
    const bones = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.3 }),
    );
    bones.visible = showSkeleton;
    
    const joints = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ color: 0x60a5fa, size: 0.03, transparent: true, opacity: 0.5 }),
    );
    joints.visible = showSkeleton;
    
    scene.add(bones, joints);

    stateRef.current = { renderer, scene, camera, bones, joints, humanoid: null, lights };
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
  }, [showSkeleton]);

  // Animate the avatar
  useEffect(() => {
    const state = stateRef.current;
    if (!state || !rig || !animation || animation.frames.length === 0) return;

    // Create humanoid mesh once
    if (!humanoidCreated.current && !state.humanoid) {
      const humanoid = createHumanoidMesh(rig);
      state.scene.add(humanoid.body);
      state.humanoid = humanoid;
      humanoidCreated.current = true;
    }

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
      const frame = animation.frames[frameIndex];
      const world = forwardKinematics(rig, frame);

      // Update skeleton (for debug mode)
      if (showSkeleton) {
        bonePairs.forEach(([parent, child], i) => {
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
      }

      // Update humanoid mesh
      if (state.humanoid) {
        // Convert world positions to flat array
        const flatPositions = new Float32Array(rig.joints.length * 3);
        world.forEach((p, i) => {
          flatPositions[i * 3] = p[0];
          flatPositions[i * 3 + 1] = p[1];
          flatPositions[i * 3 + 2] = p[2];
        });
        
        updateHumanoidMeshFromSkeleton(state.humanoid, flatPositions, rig);
        
        // Apply facial expressions
        if (frame.face) {
          applyFacialExpression(state.humanoid.head, frame.face);
        }
      }

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
  }, [rig, animation, onGlossChange, showSkeleton]);

  return <div ref={mountRef} className="avatar" />;
}
