const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const loadingEl = document.getElementById('loading');
const objectsList = document.getElementById('detected-objects');
const leftEyeBar = document.getElementById('left-eye-bar');
const rightEyeBar = document.getElementById('right-eye-bar');
const mouthBar = document.getElementById('mouth-bar');
const leftEyeVal = document.getElementById('left-eye-val');
const rightEyeVal = document.getElementById('right-eye-val');
const mouthVal = document.getElementById('mouth-val');
const faceCountEl = document.getElementById('face-count');
const happinessBar = document.getElementById('happiness-bar');
const happinessVal = document.getElementById('happiness-val');
const modeBtns = document.querySelectorAll('.mode-btn');
const toggleBtn = document.getElementById('toggle-btn');

let objectModel = null;
let extraModel = null;
let extraModelLabels = [];
let weaponModel = null;
let faceModel = null;
let isRunning = false;
let isPaused = false;
let animFrameId = null;
let confThreshold = 0.15;
let frameSkip = 0;
const WEAPON_KEYWORDS = ['gun', 'revolver', 'rifle', 'shotgun', 'pistol', 'weapon', 'knife', 'switchblade',
    'assault', 'firearm', 'sniper', 'carbine', 'submachine', 'muzzle', 'dagger', 'sword', 'bayonet',
    'shuriken', 'crossbow', 'blowgun', 'sling', 'harpoon', 'lance', 'spear', 'missile', 'warplane',
    'tank', 'cannon', 'howitzer', 'mortar', 'grenade', 'bomb', 'torpedo', 'mine', 'booby trap',
    'brass knuckles', 'baton', 'club', 'hammer', 'axe', 'hatchet', 'crowbar', 'chain saw',
    'machete', 'cleaver', 'rapier', 'sabre', 'cutlass', 'pike', 'halberd', 'flail', 'whip',
    'nunchaku', 'tonfa', 'sai', 'kama', 'shuriken', 'throwing star', 'boomerang',
    'projectile', 'artillery', 'ammunition', 'bullet', 'shell', 'rocket', 'missile launcher'];

const WEAPON_MULTIPLIER = 3;
const ALARM_THRESHOLD = 50;
const ALARM_COUNT = 3;
const ALARM_WINDOW_MS = 15000;
const ALARM_COOLDOWN_MS = 20000;
let weaponHistory = [];
let alarmActive = false;
let alarmAudioCtx = null;
let alarmOsc = null;
let alarmGain = null;
let alarmStopTimer = null;
let lastAlarmPersonTime = 0;
let detectionMode = 'crime';
let happinessHistory = [];

const HAPPINESS_THRESHOLD = 50;
const HAPPINESS_COOLDOWN_MS = 6000;

const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];
const MOUTH_UPPER = 13;
const MOUTH_LOWER = 14;
const MOUTH_LEFT = 61;
const MOUTH_RIGHT = 291;

function dist(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
}

function getPt(landmarks, i) {
    const p = landmarks[i];
    if (p == null) return [0, 0];
    if (Array.isArray(p)) return p;
    if (p.x != null) return [p.x, p.y];
    if (landmarks.length > i * 3 + 2) return [landmarks[i * 3], landmarks[i * 3 + 1]];
    return [0, 0];
}

function ear(landmarks, indices) {
    const pts = indices.map(i => getPt(landmarks, i));
    return (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / (2 * dist(pts[0], pts[3]) + 1e-6);
}

function mar(landmarks) {
    const u = getPt(landmarks, MOUTH_UPPER);
    const l = getPt(landmarks, MOUTH_LOWER);
    const lt = getPt(landmarks, MOUTH_LEFT);
    const rt = getPt(landmarks, MOUTH_RIGHT);
    return dist(u, l) / (dist(lt, rt) + 1e-6);
}

function earToPct(e) {
    const min = 0.08, max = 0.38;
    return Math.min(100, Math.max(0, ((e - min) / (max - min)) * 100));
}

function marToPct(m) {
    const min = 0.02, max = 0.6;
    return Math.min(100, Math.max(0, ((m - min) / (max - min)) * 100));
}

function happinessScore(mesh) {
    const faceW = dist(getPt(mesh, 33), getPt(mesh, 263));
    if (faceW < 1) return 0;
    const mouthW = dist(getPt(mesh, 61), getPt(mesh, 291));
    const smileRatio = mouthW / faceW;
    const smileScore = Math.min(100, Math.max(0, ((smileRatio - 0.40) / 0.25) * 100));
    const cornerMidY = (getPt(mesh, 61)[1] + getPt(mesh, 291)[1]) / 2;
    const lipCenterY = getPt(mesh, 13)[1];
    const lift = (lipCenterY - cornerMidY) / faceW;
    const liftScore = Math.min(100, Math.max(0, (lift / 0.08) * 100));
    const earL = ear(mesh, LEFT_EYE_IDX);
    const earR = ear(mesh, RIGHT_EYE_IDX);
    const avgEAR = (earL + earR) / 2;
    const squintScore = Math.min(100, Math.max(0, ((0.30 - avgEAR) / 0.15) * 100));
    const mouthOpenPct = marToPct(mar(mesh));
    if (mouthOpenPct < 20) return 0;
    const mouthHappyScore = Math.min(100, Math.max(0, 100 - Math.abs(mouthOpenPct - 20) * 5));
    return smileScore * 0.35 + liftScore * 0.25 + squintScore * 0.25 + mouthHappyScore * 0.15;
}

async function init() {
    statusEl.textContent = 'Cargando TensorFlow.js...';
    await tf.ready();
    statusEl.textContent = 'Cargando modelo de objetos (COCO-SSD)...';

    objectModel = await cocoSsd.load();
    statusEl.textContent = 'Cargando modelo facial (Facemesh)...';

    faceModel = await facemesh.load({ maxFaces: 3 });

    statusEl.textContent = 'Cargando clasificador de armas (MobileNet)...';
    weaponModel = await mobilenet.load();
    console.log(`  MobileNet: OK (1000 clases de ImageNet, detecta armas)`);

    console.log('%c✅ Modelos cargados correctamente', 'color:#00ff88;font-weight:bold');
    console.log(`  COCO-SSD: ${objectModel ? 'OK' : 'FAIL'}`);
    console.log(`  Facemesh: ${faceModel ? 'OK' : 'FAIL'}`);
    console.log(`  MobileNet: ${weaponModel ? 'OK' : 'FAIL'}`);

    try {
        statusEl.textContent = 'Cargando modelo extra (OpenImages)...';
        const raw = await fetch('https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/oid_v4_label_map.pbtxt');
        const pbtxt = await raw.text();
        extraModelLabels = parseLabelMap(pbtxt);
        console.log(`  OpenImages labels: ${extraModelLabels.length} clases cargadas`);

        extraModel = await tf.loadGraphModel(
            'https://tfhub.dev/google/tfjs-model/openimages_v4/ssd/mobilenet_v2/1/default/1',
            { fromTFHub: true }
        );
        console.log('%c✅ OpenImages SSD cargado desde TF Hub', 'color:#00ff88');
    } catch (e) {
        console.warn('%c⚠️  Modelo extra no disponible: ' + e.message, 'color:#ffaa00');
        console.log('%c💡 Para detectar armas se necesita un modelo con esas clases. Alternativas:', 'color:#88ccff');
        console.log('  1. Entrenar modelo custom con TensorFlow.js');
        console.log('  2. Usar un modelo server-side (API)');
        console.log('  3. Probar otro modelo TF.js desde TF Hub editando la URL en app.js');
        extraModel = null;
    }

    startCamera();
}

async function detectLoop() {
    if (!isRunning) return;

    if (!isPaused) {
        try {
            const detections = await detectFrame();
            drawFrame(detections.objects, detections.faces);
            updateUI(detections.objects, detections.faces);
            consoleLogStats(detections.objects, detections.faces);
        } catch (err) {
            console.error(err);
        }
    }
    frameSkip++;

    animFrameId = requestAnimationFrame(detectLoop);
}

function parseLabelMap(pbtxt) {
    const labels = [];
    const regex = /item\s*\{[^}]*?id:\s*(\d+)[^}]*?display_name:\s*"([^"]+)"/g;
    let match;
    while ((match = regex.exec(pbtxt)) !== null) {
        labels[parseInt(match[1])] = match[2];
    }
    return labels;
}

async function detectFrame() {
    let objects = [], faces = [];
    try {
        [objects, faces] = await Promise.all([
            objectModel.detect(video, 40, confThreshold),
            faceModel.estimateFaces(video)
        ]);
    } catch (err) {
        console.warn('Error en detección:', err);
    }

    if (extraModel) {
        try {
            const tensor = tf.browser.fromPixels(video).expandDims(0).toFloat();
            const input = tf.image.resizeBilinear(tensor, [320, 320]);
            const output = await extraModel.executeAsync(input);
            const [boxes, scores, classes, num] = output;
            const n = (await num.data())[0];
            const s = await scores.data();
            const c = await classes.data();
            const b = await boxes.data();
            for (let i = 0; i < n && i < 20; i++) {
                if (s[i] < confThreshold) continue;
                const label = extraModelLabels[c[i]] || 'class_' + c[i];
                const [ymin, xmin, ymax, xmax] = [b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]];
                objects.push({
                    class: label,
                    score: s[i],
                    bbox: [xmin * canvas.width, ymin * canvas.height, (xmax - xmin) * canvas.width, (ymax - ymin) * canvas.height]
                });
            }
            tensor.dispose();
            input.dispose();
            output.forEach(t => t.dispose());
        } catch (e) {
            console.warn('Error en modelo extra:', e);
        }
    }

    if (detectionMode === 'crime' && weaponModel && frameSkip % 5 === 0) {
        try {
            const predictions = await weaponModel.classify(video, 10);
            const weapons = predictions.filter(p =>
                WEAPON_KEYWORDS.some(kw => p.className.toLowerCase().includes(kw))
            );
            if (weapons.length > 0) {
                const hasPerson = objects.some(o => o.class === 'person');
                let hasWeaponAbove50 = false;

                weapons.forEach(w => {
                    const rawScore = w.probability;
                    const multScore = Math.min(1, rawScore * WEAPON_MULTIPLIER);
                    objects.push({
                        class: '⚠️ ARMA: ' + w.className.split(',')[0].trim(),
                        score: multScore,
                        bbox: [10, 10 + objects.filter(o => o.class.startsWith('⚠️')).length * 26, 200, 20]
                    });
                    if (multScore * 100 > ALARM_THRESHOLD && hasPerson) {
                        hasWeaponAbove50 = true;
                    }
                });

                checkWeaponAlarm(hasWeaponAbove50, hasPerson);

                const top = weapons.sort((a,b) => b.probability - a.probability)[0];
                const finalScore = Math.min(1, top.probability * WEAPON_MULTIPLIER) * 100;
                console.log(`%c🔫 ${top.className.split(',')[0].trim()} (x3=${finalScore.toFixed(0)}%)`, 'color:#ff4444;font-weight:bold;font-size:13px');
            }
            predictions.forEach(t => t.dispose ? t.dispose() : null);
        } catch (e) { /* silent */ }
    }

    if (detectionMode === 'happiness' && faces.length > 0 && frameSkip % 5 === 0) {
        const mesh = faces[0].scaledMesh || faces[0].mesh;
        if (mesh && mesh.length >= 468) {
            const happy = happinessScore(mesh);
            const triggered = happy > HAPPINESS_THRESHOLD;
            checkHappinessAlarm(triggered);
            if (triggered) {
                console.log(`%c😊 Felicidad: ${happy.toFixed(0)}% [${happinessHistory.length}/4 en 2s]`, 'color:#ffdd00;font-weight:bold;font-size:13px');
            }
        }
    }

    if (objects.length > 0) {
        const real = objects.filter(o => !o.class.startsWith('⚠️'));
        if (real.length > 0) console.log(`[DETECT] ${real.length} objeto(s):`, real.map(o => `${o.class} ${(o.score*100).toFixed(0)}%`).join(', '));
    }
    return { objects, faces };
}

function drawFrame(objects, faces) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    objects.forEach(obj => {
        const [x, y, w, h] = obj.bbox;
        const isWeapon = obj.class.startsWith('⚠️');

        ctx.strokeStyle = isWeapon ? '#ff2222' : '#00ff88';
        ctx.lineWidth = isWeapon ? 3 : 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = isWeapon ? 'rgba(255,0,0,0.25)' : 'rgba(0,255,136,0.15)';
        ctx.fillRect(x, y, w, h);

        const label = `${obj.class} ${(obj.score * 100).toFixed(0)}%`;
        ctx.fillStyle = isWeapon ? '#ff4444' : '#00ff88';
        ctx.font = isWeapon ? 'bold 15px monospace' : 'bold 14px monospace';
        ctx.fillText(label, x, y - 6);
    });

    faces.forEach((face, fi) => {
        const mesh = face.scaledMesh || face.mesh;
        if (!mesh || mesh.length < 468) return;

        const color = fi === 0 ? '#ff44ff' : '#ff8844';

        mesh.forEach((_, i) => {
            const p = getPt(mesh, i);
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.beginPath();
            ctx.arc(p[0], p[1], 1.2, 0, Math.PI * 2);
            ctx.fill();
        });

        drawEyeContour(ctx, mesh, LEFT_EYE_IDX, color);
        drawEyeContour(ctx, mesh, RIGHT_EYE_IDX, color);
        drawMouthOutline(ctx, mesh, '#ffdd44');

        if (fi === 0) {
            const earL = ear(mesh, LEFT_EYE_IDX);
            const earR = ear(mesh, RIGHT_EYE_IDX);
            const marV = mar(mesh);

            ctx.fillStyle = '#ff44ff';
            ctx.font = '11px monospace';
            ctx.fillText(`O.I: ${earToPct(earL).toFixed(0)}%`, getPt(mesh, 33)[0], getPt(mesh, 33)[1] - 12);
            ctx.fillText(`O.D: ${earToPct(earR).toFixed(0)}%`, getPt(mesh, 362)[0], getPt(mesh, 362)[1] - 12);
            ctx.fillText(`Boca: ${marToPct(marV).toFixed(0)}%`, getPt(mesh, 13)[0] - 20, getPt(mesh, 14)[1] + 18);
            const happy = happinessScore(mesh);
            ctx.fillStyle = '#ffdd00';
            ctx.font = 'bold 11px monospace';
            ctx.fillText(`😊 ${happy.toFixed(0)}%`, getPt(mesh, 13)[0] + 60, getPt(mesh, 14)[1] + 18);
        }
    });
}

function drawEyeContour(ctx, mesh, indices, color) {
    const pts = indices.map(i => getPt(mesh, i));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    ctx.lineTo(pts[1][0], pts[1][1]);
    ctx.lineTo(pts[2][0], pts[2][1]);
    ctx.lineTo(pts[3][0], pts[3][1]);
    ctx.lineTo(pts[4][0], pts[4][1]);
    ctx.lineTo(pts[5][0], pts[5][1]);
    ctx.closePath();
    ctx.stroke();
}

function drawMouthOutline(ctx, mesh, color) {
    const outer = getPt(mesh, MOUTH_LEFT);
    const upper = getPt(mesh, MOUTH_UPPER);
    const r = getPt(mesh, MOUTH_RIGHT);
    const lower = getPt(mesh, MOUTH_LOWER);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(outer[0], outer[1]);
    ctx.quadraticCurveTo(
        (outer[0] + r[0]) / 2, upper[1],
        r[0], r[1]
    );
    ctx.quadraticCurveTo(
        (outer[0] + r[0]) / 2, lower[1],
        outer[0], outer[1]
    );
    ctx.stroke();
}

function updateUI(objects, faces) {
    objectsList.innerHTML = '';
    const realObjs = objects.filter(o => !o.class.startsWith('⚠️'));
    const weapons = objects.filter(o => o.class.startsWith('⚠️'));
    if (realObjs.length === 0 && weapons.length === 0) {
        objectsList.innerHTML = '<li class="empty">Sin objetos detectados</li>';
    } else {
        weapons.forEach(obj => {
            const li = document.createElement('li');
            li.style.color = '#ff4444';
            li.style.fontWeight = 'bold';
            li.innerHTML = `<span class="name">${obj.class}</span><span class="conf" style="color:#ff6666">${(obj.score * 100).toFixed(0)}%</span>`;
            objectsList.appendChild(li);
        });
        if (weapons.length > 0 && realObjs.length > 0) {
            const sep = document.createElement('li');
            sep.style.borderTop = '1px solid #333';
            sep.style.margin = '2px 0';
            objectsList.appendChild(sep);
        }
        realObjs.forEach(obj => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="name">${obj.class}</span><span class="conf">${(obj.score * 100).toFixed(0)}%</span>`;
            objectsList.appendChild(li);
        });
    }

    if (faces.length > 0) {
        const mesh = faces[0].scaledMesh || faces[0].mesh;
        if (mesh && mesh.length >= 468) {
            const earL = ear(mesh, LEFT_EYE_IDX);
            const earR = ear(mesh, RIGHT_EYE_IDX);
            const marV = mar(mesh);
            const pctL = earToPct(earL);
            const pctR = earToPct(earR);
            const pctM = marToPct(marV);
            const happy = happinessScore(mesh);

            leftEyeBar.style.width = pctL + '%';
            rightEyeBar.style.width = pctR + '%';
            mouthBar.style.width = pctM + '%';
            happinessBar.style.width = happy.toFixed(0) + '%';
            leftEyeVal.textContent = pctL.toFixed(0) + '%';
            rightEyeVal.textContent = pctR.toFixed(0) + '%';
            mouthVal.textContent = pctM.toFixed(0) + '%';
            happinessVal.textContent = happy.toFixed(0) + '%';
        }
    } else {
        leftEyeBar.style.width = '0%';
        rightEyeBar.style.width = '0%';
        mouthBar.style.width = '0%';
        happinessBar.style.width = '0%';
        leftEyeVal.textContent = '0%';
        rightEyeVal.textContent = '0%';
        mouthVal.textContent = '0%';
        happinessVal.textContent = '0%';
    }

    faceCountEl.textContent = `Rostros detectados: ${faces.length}`;
}

let logCounter = 0;

function consoleLogStats(objects, faces) {
    if (++logCounter % 10 !== 0) return;
    const confPct = (confThreshold * 100).toFixed(0);

    console.clear();
    console.log('%c=== CamAI - Detección en Tiempo Real ===', 'font-weight:bold;font-size:14px');
    console.log(`%cUmbral confianza: ${confPct}%  |  Objetos: ${objects.length}  |  Rostros: ${faces.length}`, 'color:#888');
    console.log('');

    console.log(`%c📷 Objetos detectados: ${objects.length}`, 'font-weight:bold');
    if (objects.length > 0) {
        console.table(objects.map(o => ({
            Objeto: o.class,
            Confianza: (o.score * 100).toFixed(1) + '%'
        })));
    } else {
        console.log('  (ninguno — probá bajar el slider de confianza)');
    }

    console.log('');
    console.log(`%c👤 Rostros detectados: ${faces.length}`, 'font-weight:bold');

    faces.forEach((face, fi) => {
        const mesh = face.scaledMesh || face.mesh;
        if (!mesh || mesh.length < 468) return;

        const pctL = earToPct(ear(mesh, LEFT_EYE_IDX));
        const pctR = earToPct(ear(mesh, RIGHT_EYE_IDX));
        const pctM = marToPct(mar(mesh));
        const happy = happinessScore(mesh);

        console.log(`  ── Persona #${fi + 1} ──`);
        console.log(`  👁  Ojo izquierdo: ${pctL.toFixed(1)}% abierto`);
        console.log(`  👁  Ojo derecho:   ${pctR.toFixed(1)}% abierto`);
        console.log(`  👄  Boca:         ${pctM.toFixed(1)}% abierta`);
        console.log(`  😊  Felicidad:    ${happy.toFixed(1)}%`);
        console.log('');
    });
}

function startAlarm() {
    if (alarmActive) return;
    if (detectionMode === 'happiness') { startHappyAlarm(); return; }
    alarmActive = true;
    document.getElementById('alarm-bar').classList.add('active');
    document.getElementById('alarm-bar').textContent = '🚨 CRIMEN';

    alarmAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    alarmOsc = alarmAudioCtx.createOscillator();
    alarmGain = alarmAudioCtx.createGain();
    alarmOsc.type = 'sawtooth';
    alarmOsc.frequency.value = 880;
    alarmGain.gain.value = 0.3;
    alarmOsc.connect(alarmGain);
    alarmGain.connect(alarmAudioCtx.destination);
    alarmOsc.start();

    let toggle = false;
    const pulse = setInterval(() => {
        if (!alarmActive) { clearInterval(pulse); return; }
        alarmOsc.frequency.value = toggle ? 660 : 880;
        toggle = !toggle;
    }, 400);

    alarmOsc._pulseInterval = pulse;
    console.log('%c🚨 ALARMA ACTIVADA - ARMA DETECTADA', 'color:#ff0000;font-weight:bold;font-size:16px');
}

function startHappyAlarm() {
    alarmActive = true;
    document.getElementById('alarm-bar').classList.add('active');
    document.getElementById('alarm-bar').classList.add('happy');
    document.getElementById('alarm-bar').textContent = '😊 FELICIDAD';

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.value = 523.25;
    gain.gain.value = 0.12;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();

    const notes = [523.25, 659.25, 783.99, 1046.50];
    let noteIdx = 0;
    const pulse = setInterval(() => {
        if (!alarmActive) { clearInterval(pulse); return; }
        osc.frequency.setValueAtTime(notes[noteIdx], ctx.currentTime);
        noteIdx = (noteIdx + 1) % notes.length;
    }, 350);

    osc._pulseInterval = pulse;
    alarmAudioCtx = ctx;
    alarmOsc = osc;
    alarmGain = gain;
    console.log('%c😊 FELICIDAD DETECTADA', 'color:#ffdd00;font-weight:bold;font-size:16px');
}

function stopAlarm() {
    if (!alarmActive) return;
    alarmActive = false;
    document.getElementById('alarm-bar').classList.remove('active');
    document.getElementById('alarm-bar').classList.remove('happy');
    document.getElementById('alarm-bar').textContent = '🚨 INACTIVO';

    if (alarmOsc) {
        clearInterval(alarmOsc._pulseInterval);
        alarmOsc.stop();
        alarmOsc.disconnect();
        alarmOsc = null;
    }
    if (alarmGain) { alarmGain.disconnect(); alarmGain = null; }
    if (alarmAudioCtx) { alarmAudioCtx.close(); alarmAudioCtx = null; }
    if (alarmStopTimer) { clearTimeout(alarmStopTimer); alarmStopTimer = null; }
    console.log('%c✅ ALARMA DESACTIVADA', 'color:#00ff88;font-weight:bold');
}

function checkWeaponAlarm(hasWeaponAbove50, hasPerson) {
    const now = Date.now();

    if (hasWeaponAbove50 && hasPerson) {
        lastAlarmPersonTime = now;
        weaponHistory.push(now);

        weaponHistory = weaponHistory.filter(t => now - t < ALARM_WINDOW_MS);

        if (!alarmActive && weaponHistory.length >= ALARM_COUNT) {
            startAlarm();
        }
    }

    if (alarmActive) {
        if (alarmStopTimer) clearTimeout(alarmStopTimer);
        alarmStopTimer = setTimeout(() => {
            if (now - lastAlarmPersonTime >= ALARM_COOLDOWN_MS) {
                stopAlarm();
            }
        }, ALARM_COOLDOWN_MS);

        if (now - lastAlarmPersonTime >= ALARM_COOLDOWN_MS) {
            stopAlarm();
        }
    }
}

function checkHappinessAlarm(triggered) {
    const now = Date.now();
    if (triggered) {
        happinessHistory.push(now);
        happinessHistory = happinessHistory.filter(t => now - t < 2000);
        lastAlarmPersonTime = now;

        if (!alarmActive && happinessHistory.length >= 4) {
            let valid = true;
            for (let i = 1; i < happinessHistory.length; i++) {
                if (happinessHistory[i] - happinessHistory[i - 1] > 500) {
                    valid = false;
                    break;
                }
            }
            if (valid) startAlarm();
        }
}
    if (alarmActive) {
        if (alarmStopTimer) clearTimeout(alarmStopTimer);
        alarmStopTimer = setTimeout(() => {
            if (now - lastAlarmPersonTime >= HAPPINESS_COOLDOWN_MS) {
                stopAlarm();
            }
        }, HAPPINESS_COOLDOWN_MS);
        if (now - lastAlarmPersonTime >= HAPPINESS_COOLDOWN_MS) {
            stopAlarm();
        }
    }
}

function setMode(mode) {
    detectionMode = mode;
    weaponHistory = [];
    happinessHistory = [];
    if (alarmActive) stopAlarm();
    modeBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
    document.getElementById('alarm-bar').textContent = '🚨 INACTIVO';
    console.log(`%c🔄 Modo: ${mode === 'crime' ? '🚨 Crimen' : '😊 Felicidad'}`, 'color:#88ccff;font-weight:bold');
}

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    video.removeAttribute('src');
    video.removeAttribute('srcObject');
}

function loadVideo(event) {
    const file = event.target.files[0];
    if (!file) return;

    stopCamera();
    isPaused = false;
    toggleBtn.textContent = 'Pausar';
    toggleBtn.classList.remove('paused');

    const url = URL.createObjectURL(file);
    video.src = url;
    video.loop = false;

    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        video.play();
        loadingEl.classList.add('hidden');
        isRunning = true;
        statusEl.textContent = 'Procesando...';
        document.getElementById('source-indicator').textContent = `🎬 ${file.name}`;
    };

    video.onended = () => {
        URL.revokeObjectURL(url);
        document.getElementById('source-indicator').textContent = '📷 Cámara';
        startCamera();
    };
}

function startCamera() {
    statusEl.textContent = 'Iniciando cámara...';
    navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
    }).then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            loadingEl.classList.add('hidden');
            isRunning = true;
            statusEl.textContent = 'Procesando...';
            if (!animFrameId) detectLoop();
        };
    }).catch(err => {
        statusEl.textContent = 'Error cámara: ' + err.message;
    });
}

function setConf(val) {
    confThreshold = val / 100;
    document.getElementById('conf-val').textContent = val + '%';
}

function toggleDetection() {
    isPaused = !isPaused;
    toggleBtn.textContent = isPaused ? 'Reanudar' : 'Pausar';
    toggleBtn.classList.toggle('paused', isPaused);
}

document.addEventListener('DOMContentLoaded', init);
