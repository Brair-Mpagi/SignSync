/**
 * Cartoon humanoid avatar for sign language.
 * 
 * Creates a cohesive, recognizable human character with:
 * - Connected head, neck, torso forming one unified body
 * - Natural body proportions and smooth transitions
 * - Detailed hands with five clearly visible fingers
 * - Proper anatomical structure for sign language articulation
 */

import * as THREE from 'three';
import type { Rig } from '../api';

export interface HumanoidParts {
  body: THREE.Group;
  head: THREE.Group;
  torso: THREE.Group;
  leftShoulder: THREE.Group;
  rightShoulder: THREE.Group;
  leftArm: THREE.Group;
  rightArm: THREE.Group;
  leftForearm: THREE.Group;
  rightForearm: THREE.Group;
  leftHand: THREE.Group;
  rightHand: THREE.Group;
}

/**
 * Create a properly shaped finger with natural tapering
 */
function createFinger(baseWidth: number, length1: number, length2: number, length3: number): THREE.Group {
  const finger = new THREE.Group();
  const skinMat = new THREE.MeshPhongMaterial({
    color: 0xffc4a3,
    shininess: 8,
    flatShading: false,
  });

  // Segment 1 (base) - widest
  const seg1 = new THREE.CylinderGeometry(baseWidth * 0.9, baseWidth, length1, 12);
  seg1.translate(0, length1 / 2, 0);
  const mesh1 = new THREE.Mesh(seg1, skinMat);
  finger.add(mesh1);

  // Joint 1
  const joint1 = new THREE.SphereGeometry(baseWidth * 0.95, 10, 10);
  const jointMesh1 = new THREE.Mesh(joint1, skinMat);
  jointMesh1.position.y = length1;
  finger.add(jointMesh1);

  // Segment 2 (middle) - medium
  const seg2 = new THREE.CylinderGeometry(baseWidth * 0.75, baseWidth * 0.85, length2, 12);
  seg2.translate(0, length2 / 2, 0);
  const mesh2 = new THREE.Mesh(seg2, skinMat);
  mesh2.position.y = length1;
  finger.add(mesh2);

  // Joint 2
  const joint2 = new THREE.SphereGeometry(baseWidth * 0.8, 10, 10);
  const jointMesh2 = new THREE.Mesh(joint2, skinMat);
  jointMesh2.position.y = length1 + length2;
  finger.add(jointMesh2);

  // Segment 3 (tip) - tapered with rounded end
  const seg3 = new THREE.CylinderGeometry(baseWidth * 0.5, baseWidth * 0.7, length3, 12);
  seg3.translate(0, length3 / 2, 0);
  const mesh3 = new THREE.Mesh(seg3, skinMat);
  mesh3.position.y = length1 + length2;
  finger.add(mesh3);

  // Fingertip cap
  const tip = new THREE.SphereGeometry(baseWidth * 0.55, 10, 10);
  const tipMesh = new THREE.Mesh(tip, skinMat);
  tipMesh.position.y = length1 + length2 + length3;
  tipMesh.scale.y = 0.7;
  finger.add(tipMesh);

  return finger;
}

/**
 * Create a realistic cartoon hand with proper palm and five fingers
 */
function createHand(side: 'left' | 'right'): THREE.Group {
  const hand = new THREE.Group();
  const sign = side === 'left' ? 1 : -1;
  
  const skinMat = new THREE.MeshPhongMaterial({
    color: 0xffc4a3,
    shininess: 8,
    flatShading: false,
  });

  // Palm - properly shaped
  const palmShape = new THREE.Shape();
  palmShape.moveTo(-0.04, 0);
  palmShape.lineTo(-0.035, -0.09);
  palmShape.lineTo(0.035, -0.09);
  palmShape.lineTo(0.04, 0);
  palmShape.lineTo(-0.04, 0);
  
  const palmGeo = new THREE.ExtrudeGeometry(palmShape, {
    depth: 0.025,
    bevelEnabled: true,
    bevelThickness: 0.005,
    bevelSize: 0.005,
    bevelSegments: 3,
  });
  palmGeo.rotateX(-Math.PI / 2);
  palmGeo.translate(0, -0.045, 0);
  const palm = new THREE.Mesh(palmGeo, skinMat);
  hand.add(palm);

  // Wrist - thicker connection
  const wristGeo = new THREE.CylinderGeometry(0.04, 0.042, 0.05, 16);
  const wrist = new THREE.Mesh(wristGeo, skinMat);
  wrist.position.y = 0.025;
  hand.add(wrist);

  // Finger specifications: [name, x_pos, z_pos, baseWidth, len1, len2, len3, rotation]
  const fingerSpecs: Array<[string, number, number, number, number, number, number, number]> = [
    ['thumb', 0.035 * sign, 0.015, 0.020, 0.045, 0.040, 0.030, -0.6 * sign],
    ['index', 0.022 * sign, -0.095, 0.018, 0.055, 0.045, 0.035, 0],
    ['middle', 0.007 * sign, -0.100, 0.019, 0.060, 0.048, 0.037, 0],
    ['ring', -0.008 * sign, -0.095, 0.017, 0.055, 0.042, 0.033, 0],
    ['pinky', -0.023 * sign, -0.085, 0.015, 0.045, 0.035, 0.028, 0],
  ];

  fingerSpecs.forEach(([name, x, z, width, l1, l2, l3, rot]) => {
    const finger = createFinger(width, l1, l2, l3);
    finger.name = `${side}_${name}`;
    finger.position.set(x, 0, z);
    finger.rotation.z = rot;
    if (name === 'thumb') {
      finger.rotation.x = 0.4;
    }
    hand.add(finger);
  });

  return hand;
}

/**
 * Create unified humanoid body - all parts connected naturally
 */
export function createHumanoidMesh(_rig: Rig): HumanoidParts {
  const body = new THREE.Group();
  body.name = 'humanoid';

  const skinMat = new THREE.MeshPhongMaterial({
    color: 0xffc4a3,
    shininess: 10,
    flatShading: false,
  });

  const shirtMat = new THREE.MeshPhongMaterial({
    color: 0x4a90e2,
    shininess: 15,
  });

  const pantsMat = new THREE.MeshPhongMaterial({
    color: 0x2c3e50,
    shininess: 10,
  });

  // === TORSO (unified piece) ===
  const torso = new THREE.Group();
  torso.name = 'torso';

  // Chest (wider at shoulders)
  const chestGeo = new THREE.CylinderGeometry(0.20, 0.22, 0.40, 20);
  const chest = new THREE.Mesh(chestGeo, shirtMat);
  chest.position.y = 0.20;
  torso.add(chest);

  // Waist
  const waistGeo = new THREE.CylinderGeometry(0.22, 0.20, 0.20, 20);
  const waist = new THREE.Mesh(waistGeo, shirtMat);
  waist.position.y = -0.10;
  torso.add(waist);

  // Hips
  const hipsGeo = new THREE.CylinderGeometry(0.20, 0.19, 0.18, 20);
  const hips = new THREE.Mesh(hipsGeo, pantsMat);
  hips.position.y = -0.29;
  torso.add(hips);

  body.add(torso);

  // === HEAD (connected via neck) ===
  const headGroup = new THREE.Group();
  headGroup.name = 'head';

  // Neck - THICK enough to be visible
  const neckGeo = new THREE.CylinderGeometry(0.065, 0.075, 0.15, 16);
  const neck = new THREE.Mesh(neckGeo, skinMat);
  neck.position.y = -0.075;
  headGroup.add(neck);

  // Head sphere - cartoon proportions
  const headGeo = new THREE.SphereGeometry(0.16, 32, 32);
  headGeo.scale(0.95, 1.1, 1.0); // Slightly oval
  const headMesh = new THREE.Mesh(headGeo, skinMat);
  headMesh.position.y = 0.05;
  headGroup.add(headMesh);

  // Face features
  // Eyes
  const eyeGeo = new THREE.SphereGeometry(0.022, 16, 16);
  const eyeMat = new THREE.MeshPhongMaterial({ color: 0xffffff });
  const leftEyeWhite = new THREE.Mesh(eyeGeo, eyeMat);
  leftEyeWhite.position.set(0.055, 0.08, 0.14);
  headGroup.add(leftEyeWhite);
  
  const rightEyeWhite = new THREE.Mesh(eyeGeo, eyeMat);
  rightEyeWhite.position.set(-0.055, 0.08, 0.14);
  headGroup.add(rightEyeWhite);

  // Pupils
  const pupilGeo = new THREE.SphereGeometry(0.012, 12, 12);
  const pupilMat = new THREE.MeshPhongMaterial({ color: 0x2a2a2a });
  const leftPupil = new THREE.Mesh(pupilGeo, pupilMat);
  leftPupil.position.set(0.055, 0.08, 0.158);
  headGroup.add(leftPupil);
  
  const rightPupil = new THREE.Mesh(pupilGeo, pupilMat);
  rightPupil.position.set(-0.055, 0.08, 0.158);
  headGroup.add(rightPupil);

  // Eyebrows
  const browGeo = new THREE.BoxGeometry(0.050, 0.010, 0.008);
  const browMat = new THREE.MeshPhongMaterial({ color: 0x4a3728 });
  const leftBrow = new THREE.Mesh(browGeo, browMat);
  leftBrow.position.set(0.055, 0.115, 0.14);
  leftBrow.name = 'left_brow';
  headGroup.add(leftBrow);
  
  const rightBrow = new THREE.Mesh(browGeo, browMat);
  rightBrow.position.set(-0.055, 0.115, 0.14);
  rightBrow.name = 'right_brow';
  headGroup.add(rightBrow);

  // Nose
  const noseGeo = new THREE.ConeGeometry(0.018, 0.040, 12);
  noseGeo.rotateX(Math.PI / 2);
  const nose = new THREE.Mesh(noseGeo, skinMat);
  nose.position.set(0, 0.045, 0.15);
  headGroup.add(nose);

  // Mouth
  const mouthGeo = new THREE.BoxGeometry(0.070, 0.015, 0.015);
  const mouthMat = new THREE.MeshPhongMaterial({ color: 0xcc6677 });
  const mouth = new THREE.Mesh(mouthGeo, mouthMat);
  mouth.position.set(0, 0.005, 0.145);
  mouth.name = 'mouth';
  headGroup.add(mouth);

  // Ears
  const earGeo = new THREE.SphereGeometry(0.030, 12, 12);
  earGeo.scale(0.6, 1, 0.4);
  const leftEar = new THREE.Mesh(earGeo, skinMat);
  leftEar.position.set(0.145, 0.05, 0);
  headGroup.add(leftEar);
  
  const rightEar = new THREE.Mesh(earGeo, skinMat);
  rightEar.position.set(-0.145, 0.05, 0);
  headGroup.add(rightEar);

  body.add(headGroup);

  // === SHOULDERS (attachment points for arms) ===
  const leftShoulder = new THREE.Group();
  leftShoulder.name = 'left_shoulder_group';
  const shoulderBallL = new THREE.SphereGeometry(0.055, 16, 16);
  const shoulderMeshL = new THREE.Mesh(shoulderBallL, skinMat);
  leftShoulder.add(shoulderMeshL);
  body.add(leftShoulder);

  const rightShoulder = new THREE.Group();
  rightShoulder.name = 'right_shoulder_group';
  const shoulderBallR = new THREE.SphereGeometry(0.055, 16, 16);
  const shoulderMeshR = new THREE.Mesh(shoulderBallR, skinMat);
  rightShoulder.add(shoulderMeshR);
  body.add(rightShoulder);

  // === ARMS (upper arm) ===
  const leftArm = new THREE.Group();
  leftArm.name = 'left_arm_group';
  const upperArmGeoL = new THREE.CylinderGeometry(0.045, 0.040, 0.30, 16);
  upperArmGeoL.translate(0, -0.15, 0);
  const upperArmL = new THREE.Mesh(upperArmGeoL, skinMat);
  leftArm.add(upperArmL);
  // Elbow
  const elbowL = new THREE.SphereGeometry(0.042, 14, 14);
  const elbowMeshL = new THREE.Mesh(elbowL, skinMat);
  elbowMeshL.position.y = -0.30;
  leftArm.add(elbowMeshL);
  body.add(leftArm);

  const rightArm = new THREE.Group();
  rightArm.name = 'right_arm_group';
  const upperArmGeoR = new THREE.CylinderGeometry(0.045, 0.040, 0.30, 16);
  upperArmGeoR.translate(0, -0.15, 0);
  const upperArmR = new THREE.Mesh(upperArmGeoR, skinMat);
  rightArm.add(upperArmR);
  const elbowR = new THREE.SphereGeometry(0.042, 14, 14);
  const elbowMeshR = new THREE.Mesh(elbowR, skinMat);
  elbowMeshR.position.y = -0.30;
  rightArm.add(elbowMeshR);
  body.add(rightArm);

  // === FOREARMS ===
  const leftForearm = new THREE.Group();
  leftForearm.name = 'left_forearm_group';
  const forearmGeoL = new THREE.CylinderGeometry(0.038, 0.035, 0.28, 16);
  forearmGeoL.translate(0, -0.14, 0);
  const forearmL = new THREE.Mesh(forearmGeoL, skinMat);
  leftForearm.add(forearmL);
  // Wrist joint
  const wristJointL = new THREE.SphereGeometry(0.037, 12, 12);
  const wristMeshL = new THREE.Mesh(wristJointL, skinMat);
  wristMeshL.position.y = -0.28;
  leftForearm.add(wristMeshL);
  body.add(leftForearm);

  const rightForearm = new THREE.Group();
  rightForearm.name = 'right_forearm_group';
  const forearmGeoR = new THREE.CylinderGeometry(0.038, 0.035, 0.28, 16);
  forearmGeoR.translate(0, -0.14, 0);
  const forearmR = new THREE.Mesh(forearmGeoR, skinMat);
  rightForearm.add(forearmR);
  const wristJointR = new THREE.SphereGeometry(0.037, 12, 12);
  const wristMeshR = new THREE.Mesh(wristJointR, skinMat);
  wristMeshR.position.y = -0.28;
  rightForearm.add(wristMeshR);
  body.add(rightForearm);

  // === HANDS ===
  const leftHand = createHand('left');
  leftHand.name = 'left_hand_group';
  body.add(leftHand);

  const rightHand = createHand('right');
  rightHand.name = 'right_hand_group';
  body.add(rightHand);

  return {
    body,
    head: headGroup,
    torso,
    leftShoulder,
    rightShoulder,
    leftArm,
    rightArm,
    leftForearm,
    rightForearm,
    leftHand,
    rightHand,
  };
}

/**
 * Update humanoid from skeleton - maintaining body connectivity
 */
export function updateHumanoidMeshFromSkeleton(
  parts: HumanoidParts,
  jointPositions: Float32Array,
  rig: Rig
): void {
  const jointIndex = new Map(rig.joints.map((j, i) => [j.name, i]));
  
  const getPos = (name: string): THREE.Vector3 => {
    const idx = jointIndex.get(name);
    if (idx === undefined) return new THREE.Vector3();
    return new THREE.Vector3(
      -jointPositions[idx * 3],
      -jointPositions[idx * 3 + 1],
      jointPositions[idx * 3 + 2]
    );
  };

  // Position torso at hips
  const hipsPos = getPos('hips');
  parts.torso.position.copy(hipsPos);

  // Position head at neck location (moves with body)
  const neckPos = getPos('neck');
  const headPos = getPos('head');
  parts.head.position.copy(neckPos);
  
  // Orient head naturally
  const headDir = new THREE.Vector3().subVectors(headPos, neckPos);
  if (headDir.length() > 0.01) {
    const up = new THREE.Vector3(0, 1, 0);
    const targetQuat = new THREE.Quaternion().setFromUnitVectors(up, headDir.normalize());
    parts.head.quaternion.slerp(targetQuat, 0.3); // Smooth rotation
  }

  // === LEFT ARM CHAIN ===
  const lShoulder = getPos('left_shoulder');
  const lUpperArm = getPos('left_upper_arm');
  const lForearm = getPos('left_forearm');
  const lWrist = getPos('left_wrist');

  parts.leftShoulder.position.copy(lShoulder);
  parts.leftArm.position.copy(lShoulder);
  
  // Orient upper arm
  const lArmDir = new THREE.Vector3().subVectors(lUpperArm, lShoulder);
  if (lArmDir.length() > 0.01) {
    const down = new THREE.Vector3(0, -1, 0);
    parts.leftArm.quaternion.setFromUnitVectors(down, lArmDir.normalize());
  }

  // Position and orient forearm
  parts.leftForearm.position.copy(lUpperArm);
  const lForearmDir = new THREE.Vector3().subVectors(lForearm, lUpperArm);
  if (lForearmDir.length() > 0.01) {
    const down = new THREE.Vector3(0, -1, 0);
    parts.leftForearm.quaternion.setFromUnitVectors(down, lForearmDir.normalize());
  }

  // Position and orient hand
  parts.leftHand.position.copy(lWrist);
  const lHandDir = new THREE.Vector3().subVectors(lWrist, lForearm);
  if (lHandDir.length() > 0.01) {
    const down = new THREE.Vector3(0, -1, 0);
    parts.leftHand.quaternion.setFromUnitVectors(down, lHandDir.normalize());
  }

  // === RIGHT ARM CHAIN ===
  const rShoulder = getPos('right_shoulder');
  const rUpperArm = getPos('right_upper_arm');
  const rForearm = getPos('right_forearm');
  const rWrist = getPos('right_wrist');

  parts.rightShoulder.position.copy(rShoulder);
  parts.rightArm.position.copy(rShoulder);
  
  const rArmDir = new THREE.Vector3().subVectors(rUpperArm, rShoulder);
  if (rArmDir.length() > 0.01) {
    const down = new THREE.Vector3(0, -1, 0);
    parts.rightArm.quaternion.setFromUnitVectors(down, rArmDir.normalize());
  }

  parts.rightForearm.position.copy(rUpperArm);
  const rForearmDir = new THREE.Vector3().subVectors(rForearm, rUpperArm);
  if (rForearmDir.length() > 0.01) {
    const down = new THREE.Vector3(0, -1, 0);
    parts.rightForearm.quaternion.setFromUnitVectors(down, rForearmDir.normalize());
  }

  parts.rightHand.position.copy(rWrist);
  const rHandDir = new THREE.Vector3().subVectors(rWrist, rForearm);
  if (rHandDir.length() > 0.01) {
    const down = new THREE.Vector3(0, -1, 0);
    parts.rightHand.quaternion.setFromUnitVectors(down, rHandDir.normalize());
  }

  // Update fingers
  updateFingers(parts.leftHand, jointPositions, rig, 'left');
  updateFingers(parts.rightHand, jointPositions, rig, 'right');
}

/**
 * Update finger articulation
 */
function updateFingers(
  handGroup: THREE.Group,
  jointPositions: Float32Array,
  rig: Rig,
  side: 'left' | 'right'
): void {
  const jointIndex = new Map(rig.joints.map((j, i) => [j.name, i]));
  
  const getPos = (name: string): THREE.Vector3 => {
    const idx = jointIndex.get(name);
    if (idx === undefined) return new THREE.Vector3();
    return new THREE.Vector3(
      -jointPositions[idx * 3],
      -jointPositions[idx * 3 + 1],
      jointPositions[idx * 3 + 2]
    );
  };

  const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
  
  fingers.forEach(fingerName => {
    const finger = handGroup.getObjectByName(`${side}_${fingerName}`);
    if (!finger) return;

    const base = getPos(`${side}_${fingerName}_1`);
    const mid = getPos(`${side}_${fingerName}_2`);

    // Position finger relative to hand
    const handWorldPos = new THREE.Vector3();
    handGroup.getWorldPosition(handWorldPos);
    const localBase = base.clone().sub(handWorldPos);
    
    // Apply parent hand rotation inverse to get local position
    const handWorldQuat = new THREE.Quaternion();
    handGroup.getWorldQuaternion(handWorldQuat);
    localBase.applyQuaternion(handWorldQuat.invert());
    
    finger.position.copy(localBase);

    // Orient finger
    const fingerDir = new THREE.Vector3().subVectors(mid, base);
    if (fingerDir.length() > 0.01) {
      const up = new THREE.Vector3(0, 1, 0);
      const targetQuat = new THREE.Quaternion().setFromUnitVectors(up, fingerDir.normalize());
      
      // Combine with base rotation for thumb
      if (fingerName === 'thumb') {
        const baseRot = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(0.4, 0, -0.6 * (side === 'left' ? 1 : -1))
        );
        targetQuat.premultiply(baseRot);
      }
      
      finger.quaternion.copy(targetQuat);
    }
  });
}

/**
 * Apply facial expressions
 */
export function applyFacialExpression(head: THREE.Group, faceChannels: Record<string, number>): void {
  const leftBrow = head.getObjectByName('left_brow');
  const rightBrow = head.getObjectByName('right_brow');
  const mouth = head.getObjectByName('mouth');
  
  if (!leftBrow || !rightBrow || !mouth) return;

  // Reset
  leftBrow.position.y = 0.115;
  rightBrow.position.y = 0.115;
  mouth.scale.set(1, 1, 1);

  // Brow raise
  if (faceChannels.brow_raise) {
    const raise = faceChannels.brow_raise * 0.025;
    leftBrow.position.y += raise;
    rightBrow.position.y += raise;
  }

  // Brow furrow
  if (faceChannels.brow_furrow) {
    const furrow = faceChannels.brow_furrow * 0.020;
    leftBrow.position.y -= furrow;
    rightBrow.position.y -= furrow;
  }

  // Mouth open
  if (faceChannels.mouth_open) {
    mouth.scale.y = 1 + faceChannels.mouth_open * 1.2;
  }

  // Mouth wide
  if (faceChannels.mouth_wide) {
    mouth.scale.x = 1 + faceChannels.mouth_wide * 0.5;
  }
}
