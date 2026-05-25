# Entrenamiento de Redes Neuronales en GPU
* CUDA con PyTorch en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Albonire/parcial-gpu-pytorch-colab/blob/main/Taller_CUDA_PyTorch.ipynb)

---

| | |
|---|---|
| **Parcial** | Segundo Corte |
| **Materia** | Programacion Paralela y Computacion Distribuida |
| **Profesor** | Prf. Juan Alejandro Carrillo Jaimes |
| **Integrantes** | Anderson Gonzalez & [Nombre del Companero] |
| **Fecha** | 2026-I |

---

## 0. Instrucciones Generales

### Preguntas

**1. Que diferencia hay entre un notebook en la nube (Colab) y un entorno local como el del tutorial de instalacion? Cual prefieren y por que?**

*Respuesta:* [Tu respuesta aqui]

**2. Antes de comenzar, hagan una prediccion: cuantas veces mas rapida creen que sera la GPU comparada con la CPU en el entrenamiento?**

*Respuesta:* [Tu respuesta aqui]

---

## 1. Configurar el Entorno en Google Colab

**[PANTALLAZO: Verificar GPU disponible (Debe mostrar: GPU disponible: True y el nombre de la GPU)]**

> Inserta tu captura aqui: `![GPU disponible](capturas/1_gpu_disponible.png)`

**[PANTALLAZO: Estado de la GPU con nvidia-smi (Identificar: nombre de la GPU, version de CUDA, memoria total)]**

> Inserta tu captura aqui: `![nvidia-smi](capturas/1_nvidia_smi.png)`

### Preguntas

**1. La salida de `nvidia-smi` muestra campos como *Driver Version*, *Memory Usage* y *GPU-Util*. Que indica cada uno?**

*Respuesta:* [Tu respuesta aqui]

**2. Cuando activan el acelerador en Colab, que creen que ocurre fisicamente? La GPU esta en su computador o en otro lugar? Propongan una analogia con algo de la vida cotidiana.**

*Respuesta:* [Tu respuesta aqui]

**3. `torch.cuda.is_available()` retorna `True` o `False`. Que condiciones deben cumplirse para que retorne `True`? Listen al menos tres requisitos.**

*Respuesta:* [Tu respuesta aqui]

---

## 2. Conceptos: CPU vs GPU en PyTorch

**[PANTALLAZO: Salida de tensores en CPU vs GPU y Definir dispositivo]**

> Inserta tu captura aqui: `![Tensores CPU GPU](capturas/2_tensores.png)`

### Preguntas

**1. En el tutorial anterior usaron `cudaMemcpy` para mover datos entre CPU y GPU. En PyTorch eso se hace con `.to('cuda')`. Que ventaja le ven a la forma de PyTorch? Que se pierde al abstraerlo tanto?**

*Respuesta:* [Tu respuesta aqui]

**2. Diagramen en Excalidraw el flujo de un tensor desde que se crea en CPU hasta que se opera en GPU y el resultado vuelve a CPU. Etiqueten cada flecha con la operacion de PyTorch correspondiente.**

> Inserta tu diagrama Excalidraw aqui: `![Flujo tensor](capturas/2_diagrama_tensor.png)`

**3. Por que es una buena practica usar la variable `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` en lugar de escribir `'cuda'` directamente en el codigo?**

*Respuesta:* [Tu respuesta aqui]

---

## 3. Preparar los Datos: Dataset MNIST

**[PANTALLAZO: Salida de conteos de imagenes]**

> Inserta tu captura aqui: `![Conteos MNIST](capturas/3_conteos.png)`

**[PANTALLAZO: Cuadricula de 10 imagenes con sus etiquetas]**

> Inserta tu captura aqui: `![Muestras MNIST](capturas/3_muestras.png)`

### Preguntas

**1. El dataset se divide en 60,000 imagenes de entrenamiento y 10,000 de prueba. Por que no se entrena con todas las 70,000? Propongan una analogia con estudiar para un examen.**

*Respuesta:* [Tu respuesta aqui]

**2. El `DataLoader` carga los datos en lotes (*batches*) de 64 imagenes. Por que no se pasan todas las imagenes de una sola vez a la GPU? Relacionen su respuesta con el concepto de memoria que vieron en `nvidia-smi`.**

*Respuesta:* [Tu respuesta aqui]

**3. Cada imagen tiene forma `[1, 28, 28]`. Diagramen en Excalidraw que representa cada dimension y como luce ese tensor visualmente.**

> Inserta tu diagrama Excalidraw aqui: `![Tensor 1x28x28](capturas/3_diagrama_tensor.png)`

---

## 4. Construir la Red Neuronal

**[PANTALLAZO: Arquitectura impresa y el numero total de parametros]**

> Inserta tu captura aqui: `![Arquitectura](capturas/4_arquitectura.png)`

### Preguntas

**1. Diagramen en Excalidraw la arquitectura completa de la red: entrada -> capa 1 -> capa 2 -> salida. Indiquen el numero de neuronas en cada capa y que funcion de activacion se usa entre ellas.**

> Inserta tu diagrama Excalidraw aqui: `![Diagrama red](capturas/4_diagrama_red.png)`

**2. Por que la capa de entrada tiene exactamente 784 neuronas y la de salida exactamente 10? Que pasaria si pusieran 11 neuronas en la salida?**

*Respuesta:* [Tu respuesta aqui]

**3. Cuando hacen `modelo.to(device)`, que creen que se esta transfiriendo a la GPU? Es solo el codigo, o algo mas? Propongan una analogia con el tutorial de CUDA en C.**

*Respuesta:* [Tu respuesta aqui]

---

## 5. Entrenar el Modelo: CPU vs GPU

**[PANTALLAZO: Salida de entrenamiento CPU con los tiempos]**

> Inserta tu captura aqui: `![Entrenamiento CPU](capturas/5_cpu.png)`

**[PANTALLAZO: Salida de entrenamiento GPU con los tiempos]**

> Inserta tu captura aqui: `![Entrenamiento GPU](capturas/5_gpu.png)`

**[PANTALLAZO: Comparacion completa incluyendo nvidia-smi]**

> Inserta tu captura aqui: `![Comparacion](capturas/5_comparacion.png)`

### Preguntas

**1. Registren aqui los tiempos obtenidos. El resultado coincidio con la prediccion que hicieron en la seccion 0? Que los sorprendio?**

*Respuesta:* [Tu respuesta aqui]

**2. El entrenamiento repite el ciclo: *prediccion -> error -> ajuste de pesos*. Propongan una analogia con algo cotidiano que siga el mismo ciclo de mejora por repeticion.**

*Respuesta:* [Tu respuesta aqui]

**3. Por que creen que la GPU es mas rapida en esta tarea? Relacionen su respuesta con el concepto de hilos y bloques que vieron en el tutorial de CUDA en C.**

*Respuesta:* [Tu respuesta aqui]

### Analisis de la Curva de Aprendizaje

**[PANTALLAZO: Curvas de Aprendizaje generadas por `entrenar_con_loss`]**

> Inserta tu captura aqui: `![Curvas de aprendizaje](capturas/5_curvas.png)`

**1. Segun la escala, en que rango quedo el Loss final de su modelo? Lo consideran un buen resultado para 3 epocas? Justifiquen con base en la grafica que generaron.**

*Respuesta:* [Tu respuesta aqui]

**2. Observen en que epoca convergen las dos lineas. Que creen que pasaria si entrenaran 2 epocas mas -- el loss seguiria bajando indefinidamente o en algun punto se detendria? Que riesgo aparece si se entrena demasiado?**

*Respuesta:* [Tu respuesta aqui]

---

## 6. Evaluar y Visualizar Resultados

**[PANTALLAZO: Precision del modelo (Se espera mas del 95%)]**

> Inserta tu captura aqui: `![Precision](capturas/6_precision.png)`

**[PANTALLAZO: Cuadricula de predicciones con colores verde/rojo]**

> Inserta tu captura aqui: `![Predicciones](capturas/6_predicciones.png)`

### Preguntas

**1. Por que la precision se mide sobre datos que el modelo nunca vio durante el entrenamiento y no sobre los mismos datos con los que aprendio?**

*Respuesta:* [Tu respuesta aqui]

**2. Observen los digitos que el modelo clasifico mal. Tienen algo en comun? Por que creen que la red se equivoco en esos casos especificos?**

*Respuesta:* [Tu respuesta aqui]

**3. Si quisieran mejorar la precision del modelo, que cambiarian de la arquitectura o del entrenamiento? Propongan al menos dos modificaciones y justifiquen cada una.**

*Respuesta:* [Tu respuesta aqui]

---

## 7. Prueba tu Propio Digito

**[PANTALLAZO: Prediccion que se haya hecho correctamente]**

> Inserta tu captura aqui: `![Prediccion propia](capturas/7_prediccion.png)`

### Preguntas

**1. El modelo acerto con tu digito dibujado a mano? Si fallo, por que creen que se equivoco? Comparen su imagen con las del dataset MNIST -- se ven similares o muy diferentes?**

*Respuesta:* [Tu respuesta aqui]

**2. El preprocesamiento invierte los colores de la imagen (`ImageOps.invert`). Por que es necesario hacer eso antes de pasarla al modelo? Que pasaria si no se hiciera?**

*Respuesta:* [Tu respuesta aqui]

**3. Prueben con un digito que crean que va a fallar -- por ejemplo un 4 o un 9 escritos de forma poco convencional. Fallo? Que dice eso sobre las limitaciones del modelo entrenado solo con MNIST?**

*Respuesta:* [Tu respuesta aqui]

---

### Bonus: Que tan seguro esta el modelo?

**[PANTALLAZO: Probabilidades Softmax para ambos modelos]**

> Inserta tu captura aqui: `![Softmax](capturas/7_softmax.png)`

**1. Cual digito tiene la probabilidad mas alta en cada modelo? Coincide con la prediccion?**

*Respuesta:* [Tu respuesta aqui]

**2. El modelo esta seguro o dudando? Como lo saben mirando los porcentajes?**

*Respuesta:* [Tu respuesta aqui]

**3. Si el porcentaje mas alto es menor al 50%, confiarian en esa prediccion? Por que?**

*Respuesta:* [Tu respuesta aqui]

---

## 8. Preguntas de Reflexion y Entregables

**1. Ahora que completaron todo el taller, en que se parece PyTorch a programar en CUDA directamente y en que se diferencia? Cuando usarian uno y cuando el otro?**

*Respuesta:* [Tu respuesta aqui]

**2. Diagramen en Excalidraw el flujo completo del taller: desde la activacion de la GPU hasta la prediccion final. Usenlo como resumen visual de todo lo que hicieron.**

> Inserta tu diagrama Excalidraw aqui: `![Flujo completo](capturas/8_flujo_completo.png)`

**3. Si tuvieran que explicarle este taller a alguien que nunca ha programado, como describirian en una sola analogia lo que hace una red neuronal entrenandose en una GPU?**

*Respuesta:* [Tu respuesta aqui]
