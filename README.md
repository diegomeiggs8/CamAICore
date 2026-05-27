# CamAI

> Detección en tiempo real de objetos, armas y emociones faciales usando TensorFlow.js. 100% client-side, no requiere servidor ni backend.

---

## 🚨 Modo Crimen

Detección de armas mediante **MobileNet** (60+ keywords: revolver, rifle, knife, etc.) combinado con **COCO-SSD** para detectar personas.

| Indicador | Qué hace |
|-----------|----------|
| ⚠️ **SOSPECHOSO** | 2 detecciones de arma+persona → indicador amarillo sin sonido |
| 🚨 **ALARMA** | 3 detecciones en 15s → sirena 880/660Hz + pulso rojo |
| 🔇 Auto-stop | 20s sin detección desactiva la alarma |

```
📷 Cámara  [🚨 Crimen | 😊 Felicidad]  🚨 INACTIVO
┌──────────────────────────────────────────────┐
│  ⚠️ SOSPECHOSO                               │
│                                              │
│     [persona + revolver detectados]          │
│                                              │
├──────────────────────────────────────────────┤
│ Objetos:                         Facial:     │
│  ⚠️ ARMA: revolver 87%          👁 68%      │
│  persona 95%                     👁 62%      │
│  cell phone 42%                  👄 22%      │
│                                  😊 5%       │
└──────────────────────────────────────────────┘
```

## 😊 Modo Felicidad

Detección de sonrisa genuina usando 4 métricas de **Facemesh** (468 puntos faciales):

| Métrica | Peso | Qué mide |
|---------|------|----------|
| 😁 Anchura de sonrisa | 35% | Boca más ancha vs distancia ocular |
| ⬆️ Elevación de comisuras | 25% | Comisuras suben respecto al labio superior |
| 👀 Squint ocular | 25% | Ojos se achican (sonrisa Duchenne) |
| 👄 Apertura bucal | 15% | Boca ligeramente abierta (~20%) |

**Alarma:** 4 detecciones en 2s con gaps ≤0.5s → melodía ascendente (C5→E5→G5→C6) + pulso dorado. Auto-stop tras 6s.

```
📷 Cámara  [🚨 Crimen | 😊 Felicidad]  🚨 INACTIVO
┌──────────────────────────────────────────────┐
│                                              │
│     [persona sonriendo 👤😊]                 │
│                                              │
├──────────────────────────────────────────────┤
│ Objetos:                         Facial:     │
│  persona 95%                     👁 53%      │
│  cell phone 42%                  👄 22%      │
│                                  😊 85%      │
└──────────────────────────────────────────────┘
```

## 👤 Análisis Facial

| Indicador | Rango | Descripción |
|-----------|-------|-------------|
| 👁 Ojo izquierdo | 0–100% | EAR (Eye Aspect Ratio) normalizado |
| 👁 Ojo derecho | 0–100% | EAR normalizado |
| 👄 Boca abierta | 0–100% | MAR (Mouth Aspect Ratio) normalizado |
| 😊 Felicidad | 0–100% | Score combinado (4 métricas) |

Overlay en canvas con 468 landmarks faciales + contornos de ojos y boca.

## 📷 Fuentes de Video

- **Cámara web** — `getUserMedia` con auto-inicio
- **Archivo MP4** — selector de archivos con auto-retorno a cámara al terminar

## 🎛️ Controles

| Control | Función |
|---------|---------|
| ⏸️ Pausar / ▶️ Reanudar | Congela el procesamiento |
| 📁 Cargar MP4 | Video desde archivo |
| 🔄 Crimen / Felicidad | Cambia modo de detección |
| 🚨 Barra de alarma | Estado: inactivo / sospechoso / alarma |
| 🎚️ Confianza mín | Umbral 1–90% (default 15%) |

## 🖥️ Requisitos

- Navegador moderno (Chrome, Edge, Firefox)
- Cámara web (opcional, funciona con MP4)
- Servir via HTTP (`file://` no permite `getUserMedia`)

## 🚀 Uso

```bash
# Python
python -m http.server 8080

# Node
npx http-server .

# VS Code
# Extensión "Live Server" → click derecho en index.html
```

Abrir `http://localhost:8080`

## 🧠 Tecnologías

| Librería | Versión | Propósito |
|----------|---------|-----------|
| TensorFlow.js | 2.8.6 | Runtime de ML |
| COCO-SSD | 2.2.2 | Detección de objetos (90 clases) |
| Facemesh | 0.0.5 | Landmarks faciales (468 pts) |
| MobileNet | 2.1.0 | Clasificador ImageNet (1000 clases) |

## 📁 Estructura

```
camai/
├── index.html    # Página principal + UI
├── style.css     # Estilos dark theme + animaciones
├── app.js        # Lógica completa de detección
└── README.md     # Este archivo
```

## 💡 Posibles Mejoras

- [ ] Entrenar modelo custom con clases de armas específicas
- [ ] Más emociones: sorpresa, enojo, tristeza
- [ ] Exportar logs a CSV
- [ ] Historial de detecciones
- [ ] URL personalizada para modelo TF.js custom
