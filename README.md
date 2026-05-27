# CamAI

Una aplicación que usa la cámara de tu computadora para detectar **armas** y **sonrisas** en tiempo real.

---

## 🚨 Modo Crimen

Detecta personas armadas. Cuando encuentra a alguien con un arma:

- **⚠️ Sospechoso** — aparecerá un aviso amarillo si hay señales de alerta
- **🚨 Alarma** — sonará una sirena si confirma el peligro
- Se apaga sola a los 20 segundos si ya no hay peligro

## 😊 Modo Felicidad

Detecta cuando una persona está sonriendo de verdad. Usa estos indicadores:

- 😁 **Qué tan ancha es la sonrisa**
- ⬆️ **Si las comisuras de la boca suben**
- 👀 **Si los ojos se achican** (sonrisa genuina)
- 👄 **Si la boca está ligeramente abierta**

Cuando alguien sonríe por varios segundos seguidos, suena una **melodía alegre**.

## 👤 Análisis Facial

Muestra en pantalla:
- Qué tan abiertos están los ojos izquierdo y derecho
- Qué tan abierta está la boca
- Un porcentaje de **felicidad** de 0 a 100%

## 📷 Cómo se usa

1. Abre la aplicación en tu navegador (Chrome o Edge)
2. La cámara se enciende sola
3. Puedes elegir entre modo **Crimen** o **Felicidad**
4. También puedes **cargar un video MP4** en lugar de usar la cámara

## 🎮 Controles

| Botón | Qué hace |
|-------|----------|
| Pausar / Reanudar | Congela o reanuda la detección |
| Cargar MP4 | Usa un video en lugar de la cámara |
| Crimen / Felicidad | Cambia entre detectar armas o sonrisas |
| Barra roja/dorada | Muestra si hay alarma activa |
| Confianza mín | Regula qué tan sensible es la detección |

## 💻 Requisitos

- Un navegador como **Chrome**, **Edge** o **Firefox**
- **Cámara web** (no obligatoria, puedes usar un video MP4)
- Necesitas **iniciar un servidor local** (no funciona abriendo el archivo directamente)

## ¿Cómo lo abro?

Abre la terminal y escribe:

```
python -m http.server 8080
```

Luego ve a `http://localhost:8080` en tu navegador.

---

Hecho con TensorFlow.js, todo funciona en el navegador, sin instalar nada.
