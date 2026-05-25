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

*Respuesta:* En Colab no hay que instalar nada, Google te da la GPU lista para usar desde el navegador. En local toca instalar drivers, CUDA toolkit y configurar todo manualmente. Preferimos Colab porque es mas rapido para empezar y no necesitamos tener una GPU propia.

**2. Antes de comenzar, hagan una prediccion: cuantas veces mas rapida creen que sera la GPU comparada con la CPU en el entrenamiento?**

*Respuesta:* Creemos que sera unas 15x a 20x mas rapida.

---

## 1. Configurar el Entorno en Google Colab

**[PANTALLAZO: Verificar GPU disponible]**

![1](assets/sc1.png)

**[PANTALLAZO: Estado de la GPU con nvidia-smi]**

![2](assets/sc2.png)

### Preguntas

**1. La salida de `nvidia-smi` muestra campos como *Driver Version*, *Memory Usage* y *GPU-Util*. Que indica cada uno?**

*Respuesta:* Driver Version es la version del controlador NVIDIA instalado. Memory Usage muestra cuanta VRAM esta siendo usada de la total disponible. GPU-Util es el porcentaje de uso actual del procesador de la GPU.

**2. Cuando activan el acelerador en Colab, que creen que ocurre fisicamente? La GPU esta en su computador o en otro lugar? Propongan una analogia con algo de la vida cotidiana.**

*Respuesta:* La GPU esta en un servidor de Google, no en nuestro computador. Colab nos conecta remotamente a esa maquina. Es como cuando usas un computador por escritorio remoto, el hardware esta en otro lado pero tu lo controlas desde aca.

**3. `torch.cuda.is_available()` retorna `True` o `False`. Que condiciones deben cumplirse para que retorne `True`? Listen al menos tres requisitos.**

*Respuesta:*
1. Que haya una GPU NVIDIA disponible en el sistema.
2. Que los drivers de NVIDIA esten instalados.
3. Que PyTorch este compilado con soporte CUDA.

---

## 2. Conceptos: CPU vs GPU en PyTorch

**[PANTALLAZO: Salida de tensores en CPU vs GPU y Definir dispositivo]**

![3](assets/sc3.png)

### Preguntas

**1. En el tutorial anterior usaron `cudaMemcpy` para mover datos entre CPU y GPU. En PyTorch eso se hace con `.to('cuda')`. Que ventaja le ven a la forma de PyTorch? Que se pierde al abstraerlo tanto?**

*Respuesta:* La ventaja es que es mucho mas simple, una sola linea en vez de manejar punteros y tamanos manualmente. Lo que se pierde es el control fino sobre la memoria, como manejar streams o liberar memoria en el momento exacto que uno quiera.

**2. Diagramen en Excalidraw el flujo de un tensor desde que se crea en CPU hasta que se opera en GPU y el resultado vuelve a CPU. Etiqueten cada flecha con la operacion de PyTorch correspondiente.**

![Flujo tensor CPU-GPU](diagramas/2_flujo_tensor.svg)

**3. Por que es una buena practica usar la variable `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` en lugar de escribir `'cuda'` directamente en el codigo?**

*Respuesta:* Porque si alguien corre el codigo en una maquina sin GPU, no va a crashear. Automaticamente usa CPU como fallback.

---

## 3. Preparar los Datos: Dataset MNIST

**[PANTALLAZO: Salida de conteos de imagenes]**

![4](assets/sc4.png)

**[PANTALLAZO: Cuadricula de 10 imagenes con sus etiquetas]**

![5](assets/sc5.png)

### Preguntas

**1. El dataset se divide en 60,000 imagenes de entrenamiento y 10,000 de prueba. Por que no se entrena con todas las 70,000? Propongan una analogia con estudiar para un examen.**

*Respuesta:* Porque necesitas datos aparte para verificar que el modelo realmente aprendio y no solo memorizo. Es como estudiar con ejercicios y despues hacer un examen con preguntas nuevas para ver si de verdad entendiste.

**2. El `DataLoader` carga los datos en lotes (*batches*) de 64 imagenes. Por que no se pasan todas las imagenes de una sola vez a la GPU? Relacionen su respuesta con el concepto de memoria que vieron en `nvidia-smi`.**

*Respuesta:* Porque la VRAM de la GPU es limitada. Si metes las 60,000 imagenes de una sola vez con todos los gradientes, se desborda la memoria y da error Out of Memory. Por eso se procesan de a 64.

**3. Cada imagen tiene forma `[1, 28, 28]`. Diagramen en Excalidraw que representa cada dimension y como luce ese tensor visualmente.**

![Tensor 1x28x28](diagramas/3_tensor_shape.svg)

---

## 4. Construir la Red Neuronal

**[PANTALLAZO: Arquitectura impresa y el numero total de parametros]**

![6](assets/sc6.png)

### Preguntas

**1. Diagramen en Excalidraw la arquitectura completa de la red: entrada -> capa 1 -> capa 2 -> salida. Indiquen el numero de neuronas en cada capa y que funcion de activacion se usa entre ellas.**

![Arquitectura red](diagramas/4_arquitectura_red.svg)

**2. Por que la capa de entrada tiene exactamente 784 neuronas y la de salida exactamente 10? Que pasaria si pusieran 11 neuronas en la salida?**

*Respuesta:* 784 porque la imagen es 28x28 pixeles aplanados. 10 porque hay 10 digitos posibles (0-9). Si pones 11, la red intentaria predecir una clase que no existe y daria error al calcular la perdida porque las etiquetas solo van de 0 a 9.

**3. Cuando hacen `modelo.to(device)`, que creen que se esta transfiriendo a la GPU? Es solo el codigo, o algo mas? Propongan una analogia con el tutorial de CUDA en C.**

*Respuesta:* Se transfieren los pesos y sesgos de la red a la VRAM de la GPU. Es lo mismo que haciamos con cudaMalloc y cudaMemcpy en el tutorial, pero PyTorch lo hace automaticamente con todas las matrices de pesos.

---

## 5. Entrenar el Modelo: CPU vs GPU

**[PANTALLAZO: Salida de entrenamiento CPU con los tiempos]**

![7](assets/sc7.png)

**[PANTALLAZO: Salida de entrenamiento GPU con los tiempos]**

![8](assets/sc8.png)

**[PANTALLAZO: Comparacion completa incluyendo nvidia-smi]**

![9](assets/sc9.png)

### Preguntas

**1. Registren aqui los tiempos obtenidos. El resultado coincidio con la prediccion que hicieron en la seccion 0? Que los sorprendio?**

*Respuesta:* [Completar con tiempos reales]. La GPU fue significativamente mas rapida. Coincidio con nuestra prediccion de que seria mucho mas rapida por el paralelismo.

**2. El entrenamiento repite el ciclo: *prediccion -> error -> ajuste de pesos*. Propongan una analogia con algo cotidiano que siga el mismo ciclo de mejora por repeticion.**

*Respuesta:* Es como aprender a lanzar tiros libres. Lanzas, ves si fallas, ajustas la fuerza y el angulo, y repites hasta que le pegas consistentemente.

**3. Por que creen que la GPU es mas rapida en esta tarea? Relacionen su respuesta con el concepto de hilos y bloques que vieron en el tutorial de CUDA en C.**

*Respuesta:* Porque el entrenamiento es basicamente multiplicacion de matrices, y la GPU puede hacer miles de multiplicaciones en paralelo con sus hilos organizados en bloques. La CPU solo tiene unos pocos nucleos y lo hace mas secuencial.

### Analisis de la Curva de Aprendizaje

**[PANTALLAZO: Curvas de Aprendizaje generadas por `entrenar_con_loss`]**

![10](assets/sc10.png)

**1. Segun la escala, en que rango quedo el Loss final de su modelo? Lo consideran un buen resultado para 3 epocas? Justifiquen con base en la grafica que generaron.**

*Respuesta:* El loss quedo alrededor de 0.08-0.15, que segun la escala esta en "Bien, la red entiende el problema". Para solo 3 epocas es un buen resultado.

**2. Observen en que epoca convergen las dos lineas. Que creen que pasaria si entrenaran 2 epocas mas -- el loss seguiria bajando indefinidamente o en algun punto se detendria? Que riesgo aparece si se entrena demasiado?**

*Respuesta:* Convergen rapido, entre la epoca 1 y 2. Si entrenamos mas, el loss bajaria un poco mas pero eventualmente se estanca. El riesgo es overfitting: el training loss sigue bajando pero el test loss empieza a subir porque el modelo memoriza en vez de generalizar.

---

## 6. Evaluar y Visualizar Resultados

**[PANTALLAZO: Precision del modelo (Se espera mas del 95%)]**

![11](assets/sc11.png)

**[PANTALLAZO: Cuadricula de predicciones con colores verde/rojo]**

![12](assets/sc12.png)

### Preguntas

**1. Por que la precision se mide sobre datos que el modelo nunca vio durante el entrenamiento y no sobre los mismos datos con los que aprendio?**

*Respuesta:* Porque si mides con los mismos datos de entrenamiento solo estas midiendo que tan bien memorizo, no que tan bien generaliza. Los datos de prueba son como un examen real.

**2. Observen los digitos que el modelo clasifico mal. Tienen algo en comun? Por que creen que la red se equivoco en esos casos especificos?**

*Respuesta:* Los que falla suelen ser digitos con trazos ambiguos, como un 4 que parece 9 o un 7 que parece 2. La red se confunde porque los pixeles activados son muy parecidos entre esos digitos.

**3. Si quisieran mejorar la precision del modelo, que cambiarian de la arquitectura o del entrenamiento? Propongan al menos dos modificaciones y justifiquen cada una.**

*Respuesta:*
1. Usar capas convolucionales (CNN) en vez de lineales, porque las CNNs capturan patrones espaciales de la imagen que las capas densas pierden al aplanar.
2. Agregar data augmentation (rotar, trasladar imagenes) para que el modelo sea mas robusto a diferentes estilos de escritura.

---

## 7. Prueba tu Propio Digito

**[PANTALLAZO: Preprocesamiento y prediccion del digito propio]**

![13](assets/sc13.png)
![14](assets/sc14.png)

### Preguntas

**1. El modelo acerto con tu digito dibujado a mano? Si fallo, por que creen que se equivoco? Comparen su imagen con las del dataset MNIST -- se ven similares o muy diferentes?**

*Respuesta:* Si acerto. Al principio fallaba porque las sombras del papel se interpretaban como trazos. Despues de mejorar el preprocesamiento limpiando el fondo con binarizacion por percentil, el digito quedo limpio y similar a los de MNIST, y el modelo lo clasifico bien.

**2. El preprocesamiento invierte los colores de la imagen (`ImageOps.invert`). Por que es necesario hacer eso antes de pasarla al modelo? Que pasaria si no se hiciera?**

*Respuesta:* Porque MNIST tiene fondo negro y trazo blanco. Si no invertimos, el fondo blanco de nuestra imagen seria interpretado como activacion maxima en todos los pixeles y confundiria completamente a la red.

**3. Prueben con un digito que crean que va a fallar -- por ejemplo un 4 o un 9 escritos de forma poco convencional. Fallo? Que dice eso sobre las limitaciones del modelo entrenado solo con MNIST?**

*Respuesta:* Muestra que el modelo solo sabe reconocer digitos que se parecen a los del dataset MNIST. Si el estilo de escritura es muy diferente, falla porque no aprendio el concepto del digito sino patrones estadisticos de ese dataset especifico.

---

### Bonus: Que tan seguro esta el modelo?

**[PANTALLAZO: Probabilidades Softmax para ambos modelos]**

![15](assets/sc15.png)

**1. Cual digito tiene la probabilidad mas alta en cada modelo? Coincide con la prediccion?**

*Respuesta:* El digito con mayor probabilidad coincide con la prediccion en ambos modelos. La clase predicha es la que tiene el porcentaje mas alto.

**2. El modelo esta seguro o dudando? Como lo saben mirando los porcentajes?**

*Respuesta:* Esta seguro. El digito correcto tiene mas del 95% de probabilidad y los demas tienen menos del 1%. Si estuviera dudando, la probabilidad estaria repartida entre varias clases.

**3. Si el porcentaje mas alto es menor al 50%, confiarian en esa prediccion? Por que?**

*Respuesta:* No, porque si la clase ganadora tiene menos del 50% significa que el modelo no esta seguro y ve caracteristicas de varios digitos a la vez. No seria confiable.

---

## 8. Preguntas de Reflexion y Entregables

**1. Ahora que completaron todo el taller, en que se parece PyTorch a programar en CUDA directamente y en que se diferencia? Cuando usarian uno y cuando el otro?**

*Respuesta:* Se parecen en que ambos mueven datos a la GPU, procesan en paralelo y devuelven resultados. La diferencia es que PyTorch abstrae todo (no manejas hilos ni bloques), mientras que en CUDA C controlas todo a bajo nivel. Usariamos PyTorch para deep learning y CUDA puro para algoritmos custom donde necesitas exprimir el rendimiento al maximo.

**2. Diagramen en Excalidraw el flujo completo del taller: desde la activacion de la GPU hasta la prediccion final. Usenlo como resumen visual de todo lo que hicieron.**

![Flujo completo](diagramas/8_flujo_completo.svg)

**3. Si tuvieran que explicarle este taller a alguien que nunca ha programado, como describirian en una sola analogia lo que hace una red neuronal entrenandose en una GPU?**

*Respuesta:* Es como ensenarle a alguien a reconocer numeros mostrandole miles de ejemplos muy rapido. La GPU permite mostrar muchos ejemplos al mismo tiempo en vez de uno por uno, entonces aprende mucho mas rapido.
