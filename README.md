# Deblurring-Diffusion-IA2

**Restauración de imágenes borrosas mediante un modelo de difusión condicional.**
Proyecto final de Inteligencia Artificial II — Universidad Industrial de Santander (UIS).

Eliminamos el desenfoque de movimiento en el dataset **GoPro** con un esquema de dos etapas:
**NAFNet-64** da una restauración fiel y una **difusión residual** genera solo el detalle que le
falta. En las 1 111 imágenes de test, la difusión reduce el **LPIPS un 15 %** (0.0779 → 0.0661)
a cambio de **1.2 dB de PSNR**.

**Integrantes**

- Juan Felipe Serrano Contreras — 2230088
- María Lucía Rodríguez Olarte — 2231853
- Santiago Ardila Lagos — 2230086

---

## Resultados

Test completo del GoPro oficial: 1 111 imágenes a 1280×720.

| Método | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Imagen borrosa | 25.64 | 0.7795 | 0.2910 |
| NAFNet (pesos oficiales) | 33.71 | 0.9475 | 0.0781 |
| NAFNet-FT | 34.62 | 0.9333 | 0.0779 |
| **NAFNet-FT + difusión (nuestro)** | 33.43 | 0.9430 | **0.0661** |

La regresión (NAFNet) maximiza el PSNR, pero devuelve el *promedio* de las soluciones posibles,
que se ve suave. La difusión agrega textura y bordes: pierde algo de fidelidad píxel a píxel y
gana en percepción. Es el compromiso percepción–distorsión de Blau y Michaeli (2018).

Además, como la difusión es estocástica, con 8 muestras por imagen se obtiene un **mapa de
incertidumbre** (desviación estándar por píxel). Ese mapa se concentra en las mismas zonas que el
error real: bordes, texturas y caras.

---

## Método

```
 y (borrosa) ──► NAFNet-64 ──► ŷ ─────────────────────────────────┐
      │                         │                                 ▼
      └──── condición [y, ŷ] ───┴──► U-Net denoiser ──► r̂ ──► x̂ = ŷ + s·r̂
                                       ▲  (DDIM truncado, t₀ → 0)
                          ruido en t₀ ─┘
```

Basado en la arquitectura de doble red de Whang et al., *Deblurring via Stochastic Refinement*
(CVPR 2022).

1. **Etapa 1 · determinista.** NAFNet-64 predice ŷ ≈ E[x | y]. Se congela y sus predicciones se
   guardan en disco.
2. **Etapa 2 · estocástica.** Un DDPM condicional aprende la distribución del residuo escalado
   r = (x − ŷ)/s. El factor s lleva el residuo a desviación estándar 0.5. El residuo tiene media
   ≈ 0 y contiene sobre todo altas frecuencias, así que es más fácil de modelar que la imagen
   completa.
3. **Muestreo truncado.** No se parte de ruido puro. Se parte de ŷ con ruido en un paso
   intermedio t₀ (SDEdit / Come-Closer-Diffuse-Faster). t₀ controla cuánto detalle inventa el
   modelo y permite usar pocos pasos.

---

## Datos

[GoPro](https://seungjunnah.github.io/Datasets/gopro), versión oficial `GOPRO_Large`
(Nah et al., CVPR 2017).

- 3 214 pares borrosa / nítida, de 33 secuencias de video, a 1280×720.
- Test oficial: 1 111 pares.
- Train / val: 1 725 / 378. El split se hace **por video**, porque frames consecutivos son casi
  idénticos y mezclarlos infla la validación.

> En la entrega 2 se usó un repack de Kaggle cuyas borrosas venían de `blur_gamma` (≈ +2 niveles
> de brillo). El “+5.9 dB” del fine-tuning de entonces solo corregía ese desajuste. Desde esta
> entrega todo se mide en el GoPro oficial.

---

## Entrenamiento

| | NAFNet-64 (Etapa 1) | Difusión residual (Etapa 2) |
|---|---|---|
| Punto de partida | Pesos oficiales GoPro (TLC) | Desde cero |
| Parámetros | ≈ 68 M | ≈ 10.6 M (U-Net, canales base 64, mult. 1-2-4, 2 bloques por nivel) |
| Pérdida | PSNR loss | v-prediction + ponderación min-SNR (γ = 5) + 0.05 · L1 en frecuencia |
| Difusión | — | T = 1 000, schedule coseno, EMA 0.999 |
| Parches | 256×256, batch 2 | 128×128, 4 imágenes × 4 recortes = 16 por paso |
| Learning rate | 3e-5, warmup 500, coseno | 2e-4, warmup 1 000, coseno |
| Iteraciones | 20 000 (configurable) | 12 000 |
| Selección | Mejor PSNR en val | Mejor LPIPS en val → iteración 2 000 |

Todo se entrenó en una **GPU T4 de Google Colab**, con precisión mixta (float16; la T4 emula
bfloat16 y es unas 4 veces más lenta).

---

## Ablaciones de muestreo

Las ablaciones usan 30 imágenes de test y el mismo checkpoint, con DDIM determinista (η = 0) y
mosaicos de 256 px con solapamiento 32.

| Configuración | PSNR (dB) ↑ | LPIPS ↓ | s / imagen |
|---|---|---|---|
| NAFNet-64 + FT (sin difusión) | 33.96 | 0.0754 | — |
| t₀ = 0.25 · 20 pasos | 33.81 | 0.0668 | ≈ 14 |
| t₀ = 0.5 · 20 pasos | 33.63 | 0.0620 | 14.0 |
| t₀ = 0.75 · 20 pasos | 33.23 | 0.0568 | ≈ 14 |
| t₀ = 1.0 (ruido puro) · 20 pasos | 32.45 | 0.0571 | ≈ 14 |
| t₀ = 0.5 · 5 pasos | 33.72 | 0.0669 | 3.4 |
| **t₀ = 0.5 · 10 pasos (elegida)** | **33.66** | **0.0637** | **6.8** |
| t₀ = 0.5 · 20 pasos · promedio N = 2 | 33.78 | 0.068 | — |
| t₀ = 0.5 · 20 pasos · promedio N = 4 | 33.87 | 0.072 | — |

- **t₀** mueve el resultado por la frontera percepción–distorsión: más ruido da mejor LPIPS y
  menos PSNR. Con ruido puro ya no mejora la percepción.
- **10 pasos** conservan cerca del 87 % de la mejora de LPIPS de 20 pasos, a la mitad del tiempo.
- **Promediar N muestras** vuelve hacia NAFNet: el promedio es otra vez una imagen suave.

Configuración final: **10 pasos, t₀ = 0.5, N = 1**.

---

## Cómo reproducir

Todo el pipeline está en el notebook de Colab
**`Deblurring_GoPro_NAFNet_Difusion.ipynb`**. Al ejecutarse escribe su propio código en
`/content/work/src/` con `%%writefile`, así que no hace falta clonar nada más.

1. Sube `GOPRO_Large.zip` a tu Drive y abre el notebook en Colab con GPU (T4).
2. En la celda **Configuración** ajusta `DATASET_PATH` y `DRIVE_DIR`. Para la configuración final
   usa `SAMPLE_STEPS = 10` y `SAMPLE_T0 = 0.5`.
3. Ejecuta las etapas en orden:

| Etapa | Qué hace | Salida en `DRIVE_DIR` |
|---|---|---|
| 0 · Datos | Descomprime, crea el split por video y la caché local | `split.json` |
| 1 · NAFNet-64 | Predicciones con pesos oficiales; fine-tuning opcional (`RUN_FINETUNE`) | `nafnet/` |
| 2 · Difusión | Entrena la U-Net residual | `difusion/unet/best.pt`, `log.csv` |
| 3 · Evaluación | Ablaciones y tabla final en el test completo | `eval/ablaciones/`, `eval/final/` (`summary.md`, `per_image.csv`, `salidas/dif_unet/`) |
| 4 · Figuras | Curvas, percepción–distorsión, cualitativos e incertidumbre | `figuras/` |

Todas las etapas son **reanudables**. Los checkpoints y los CSV se escriben en Drive mientras
corren, así que si Colab se desconecta basta con volver a ejecutar la misma celda. Se necesitan
unos 6 GB de Drive además del dataset.

### Módulos que genera el notebook

| Archivo | Contenido |
|---|---|
| `common.py` | Lectura de datos, split, métricas (PSNR, SSIM, LPIPS), recortes y aumentos |
| `nafnet_arch.py` | Arquitectura NAFNet con TLC |
| `prepare.py` | Split por video y cachés en disco local |
| `predict.py` / `finetune.py` | Inferencia y fine-tuning de NAFNet-64 |
| `diffusion_lib.py` | U-Net condicional, difusión residual (v-pred, min-SNR) y muestreo DDIM truncado por mosaicos |
| `train_diffusion.py` | Entrenamiento de la Etapa 2 con EMA y selección por LPIPS |
| `evaluate.py` | Evaluación a 1280×720: borrosa vs. NAFNet vs. difusión, ablaciones |
| `figures.py` | Figuras del informe |

---

## Historia del proyecto

| Entrega | Enfoque | Resultado |
|---|---|---|
| 1 | Zero-shot (DeblurSDI): un kernel por imagen | ~15 min por imagen; supone un desenfoque uniforme |
| 2 | DDPM que genera la imagen completa desde ruido | 23.8 dB, peor que la entrada (27.7 dB) en 10 imágenes |
| 2 | NAFNet afinado en GoPro de Kaggle | +5.9 dB que en realidad corregía un desajuste gamma |
| **Final** | **NAFNet + difusión residual en el GoPro oficial** | **LPIPS −15 % en el test completo** |

El código de la entrega 2 sigue en el repositorio (`Deblurring_Difusion_Entrega_2.ipynb`,
`finetune_nafnet.py`, `checkpoints/nafnet_ft/`) como referencia.

---

## Limitaciones y trabajo futuro

- La difusión se entrenó 12 000 iteraciones por límite de GPU; más entrenamiento y un denoiser
  más grande podrían mejorar ambos ejes.
- El muestreo tarda 6.8 s por imagen en una T4; destilación o consistency models permitirían
  1–2 pasos.
- LPIPS aproxima la percepción humana; falta una evaluación con personas.

---

## Referencias

- Nah, Kim, Lee. *Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring*. CVPR 2017.
- Whang et al. *Deblurring via Stochastic Refinement*. CVPR 2022.
- Chen et al. *Simple Baselines for Image Restoration (NAFNet)*. ECCV 2022.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021.
- Meng et al. *SDEdit*. ICLR 2022 · Chung et al. *Come-Closer-Diffuse-Faster*. CVPR 2022.
- Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models* (v-prediction). ICLR 2022.
- Hang et al. *Efficient Diffusion Training via Min-SNR Weighting Strategy*. ICCV 2023.
- Blau, Michaeli. *The Perception-Distortion Tradeoff*. CVPR 2018.
- Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)*. CVPR 2018.
