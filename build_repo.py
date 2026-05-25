import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata required for Colab
nb.metadata = {
  "colab": {
    "provenance": []
  },
  "kernelspec": {
    "display_name": "Python 3",
    "name": "python3"
  },
  "language_info": {
    "name": "python"
  }
}

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("# Entrenamiento de Redes Neuronales en GPU\n* CUDA con PyTorch en Google Colab"))

# Section 1
cells.append(nbf.v4.new_markdown_cell("## 1. Configurar el Entorno en Google Colab"))
cells.append(nbf.v4.new_code_cell("""# Celda 1: Verificar GPU disponible
import torch
 
print('Version de PyTorch:', torch.__version__)
print('GPU disponible:', torch.cuda.is_available())
 
if torch.cuda.is_available():
    print('Nombre de la GPU:', torch.cuda.get_device_name(0))
    print('Memoria total (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
else:
    print('No hay GPU. Revisa que activaste el acelerador correctamente.')"""))

cells.append(nbf.v4.new_code_cell("""# Celda 2: Estado de la GPU (igual que en el tutorial de instalacion)
!nvidia-smi"""))

# Section 2
cells.append(nbf.v4.new_markdown_cell("## 2. Conceptos: CPU vs GPU en PyTorch"))
cells.append(nbf.v4.new_code_cell("""# Celda 3: Tensores en CPU vs GPU
import torch
 
# Crear un tensor en CPU (memoria del procesador)
tensor_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print('Tensor en CPU:', tensor_cpu)
print('Dispositivo:', tensor_cpu.device)   # Debe decir: cpu
 
# Mover el tensor a la GPU (memoria de la tarjeta grafica)
tensor_gpu = tensor_cpu.to('cuda')
print('Tensor en GPU:', tensor_gpu)
print('Dispositivo:', tensor_gpu.device)   # Debe decir: cuda:0
 
# Las operaciones en GPU se ejecutan en paralelo automaticamente
resultado = tensor_gpu * 2
print('Resultado (en GPU):', resultado)
print('Dispositivo resultado:', resultado.device)"""))

cells.append(nbf.v4.new_code_cell("""# Celda 4: Definir dispositivo (SIEMPRE al inicio del proyecto)
import torch
 
# Esta linea elige GPU si hay, CPU si no hay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo a usar:', device)"""))

# Section 3
cells.append(nbf.v4.new_markdown_cell("## 3. Preparar los Datos: Dataset MNIST"))
cells.append(nbf.v4.new_code_cell("""# Celda 5: Importar librerias necesarias
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
 
# Definir el dispositivo (de la celda anterior)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Usando dispositivo:', device)"""))

cells.append(nbf.v4.new_code_cell("""# Celda 6: Descargar y preparar los datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
 
# Descargar datos de entrenamiento (60,000 imagenes)
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
 
# Descargar datos de prueba (10,000 imagenes)
test_dataset  = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)
 
# DataLoader: carga los datos en lotes (batches) para entrenar de a poco
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)
 
print('Imagenes de entrenamiento:', len(train_dataset))
print('Imagenes de prueba:', len(test_dataset))
print('Lotes de entrenamiento:', len(train_loader))"""))

cells.append(nbf.v4.new_code_cell("""# Celda 7: Ver algunas imagenes del dataset
imagenes, etiquetas = next(iter(train_loader))
 
print('Forma de un lote:', imagenes.shape)  # [64, 1, 28, 28]
 
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(imagenes[i].squeeze(), cmap='gray')
    ax.set_title(f'Digito: {etiquetas[i].item()}')
    ax.axis('off')
plt.suptitle('Muestras del dataset MNIST', fontsize=14)
plt.tight_layout()
plt.show()"""))

# Section 4
cells.append(nbf.v4.new_markdown_cell("## 4. Construir la Red Neuronal"))
cells.append(nbf.v4.new_code_cell("""# Celda 8: Definir la arquitectura de la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self):
        super(RedNeuronal, self).__init__()
        # Definir las capas de la red
        self.capas = nn.Sequential(
            nn.Flatten(),          # Aplana 28x28 = 784 valores
            nn.Linear(784, 256),   # Capa 1: entrada(784) -> oculta(256)
            nn.ReLU(),             # Funcion de activacion
            nn.Linear(256, 128),   # Capa 2: oculta(256) -> oculta(128)
            nn.ReLU(),             # Funcion de activacion
            nn.Linear(128, 10),    # Capa 3: oculta(128) -> salida(10 digitos)
        )
 
    def forward(self, x):
        return self.capas(x)
 
# Crear la red y moverla a la GPU
modelo = RedNeuronal().to(device)
 
print('Arquitectura de la red:')
print(modelo)
print()
 
# Contar parametros entrenables
total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f'Total de parametros: {total_params:,}')"""))

# Section 5
cells.append(nbf.v4.new_markdown_cell("## 5. Entrenar el Modelo: CPU vs GPU"))
cells.append(nbf.v4.new_code_cell("""# Celda 9: Funcion de entrenamiento con loss (reemplazando la original)
def entrenar_con_loss(modelo, train_loader, test_loader, dispositivo, title, epocas=3):
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)

    historico_train = []
    historico_test  = []

    modelo.train()
    inicio = time.time()

    for epoca in range(epocas):
        # --- Training loss ---
        modelo.train()
        loss_train = 0
        for imagenes, etiquetas in train_loader:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            prediccion = modelo(imagenes)
            perdida = criterio(prediccion, etiquetas)

            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            loss_train += perdida.item()

        # --- Test loss ---
        modelo.eval()
        loss_test = 0
        with torch.no_grad():
            for imagenes, etiquetas in test_loader:
                imagenes = imagenes.to(dispositivo)
                etiquetas = etiquetas.to(dispositivo)
                prediccion = modelo(imagenes)
                perdida = criterio(prediccion, etiquetas)
                loss_test += perdida.item()

        avg_train = loss_train / len(train_loader)
        avg_test  = loss_test  / len(test_loader)

        historico_train.append(avg_train)
        historico_test.append(avg_test)

        print(f"Epoca {epoca+1}/{epocas} - Train loss: {avg_train:.4f} | Test loss: {avg_test:.4f}")

    tiempo = time.time() - inicio

    # --- Graficar ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epocas+1), historico_train, label='Training loss', linewidth=2)
    plt.plot(range(1, epocas+1), historico_test,  label='Test loss',     linewidth=2, linestyle='--')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.title(f'Curva de Aprendizaje {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return historico_train, historico_test, tiempo"""))

cells.append(nbf.v4.new_code_cell("""# Celda 10: Entrenamiento en CPU
print('=' * 50)
print('ENTRENAMIENTO EN CPU')
print('=' * 50)
 
# Crear nuevo modelo en CPU
modelo_cpu = RedNeuronal().to('cpu')
 
_, _, tiempo_cpu = entrenar_con_loss(modelo_cpu, train_loader, test_loader, 'cpu', 'en CPU', epocas=3)
print(f'Tiempo total en CPU: {tiempo_cpu:.2f} segundos')"""))

cells.append(nbf.v4.new_code_cell("""# Celda 11: Entrenamiento en GPU
print('=' * 50)
print('ENTRENAMIENTO EN GPU')
print('=' * 50)
 
# Crear nuevo modelo en GPU
modelo_gpu = RedNeuronal().to('cuda')
 
_, _, tiempo_gpu = entrenar_con_loss(modelo_gpu, train_loader, test_loader, 'cuda', 'en GPU', epocas=3)
print(f'Tiempo total en GPU: {tiempo_gpu:.2f} segundos')"""))

cells.append(nbf.v4.new_code_cell("""# Celda 12: Comparacion CPU vs GPU
print('\\n' + '=' * 50)
print('COMPARACION DE RENDIMIENTO')
print('=' * 50)
print(f'Tiempo en CPU:  {tiempo_cpu:.2f} segundos')
print(f'Tiempo en GPU:  {tiempo_gpu:.2f} segundos')
aceleracion = tiempo_cpu / tiempo_gpu
print(f'La GPU fue {aceleracion:.1f}x mas rapida que la CPU')
 
# Ver memoria de GPU usada durante entrenamiento
print(f'\\nMemoria GPU usada: {torch.cuda.max_memory_allocated()/1e6:.1f} MB')
!nvidia-smi"""))

# Section 6
cells.append(nbf.v4.new_markdown_cell("## 6. Evaluar y Visualizar Resultados"))
cells.append(nbf.v4.new_code_cell("""# Celda 13: Evaluar el modelo entrenado en GPU
def evaluar(modelo, loader, dispositivo):
    modelo.eval()  # Modo evaluacion (desactiva dropout, etc.)
    correctas = 0
    total = 0
 
    with torch.no_grad():  # No calcular gradientes (ahorra memoria)
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)
 
            salidas = modelo(imagenes)
            # La prediccion es la clase con mayor valor
            _, prediccion = torch.max(salidas, 1)
            total += etiquetas.size(0)
            correctas += (prediccion == etiquetas).sum().item()
 
    return 100 * correctas / total
 
precision_gpu = evaluar(modelo_gpu, test_loader, 'cuda')
precision_cpu = evaluar(modelo_cpu, test_loader, 'cpu')
print(f'Precision del modelo GPU: {precision_gpu:.2f}%')
print(f'Precision del modelo CPU: {precision_cpu:.2f}%')"""))

cells.append(nbf.v4.new_code_cell("""# Celda 14: Visualizar predicciones del modelo GPU
modelo_gpu.eval()
imagenes_test, etiquetas_test = next(iter(test_loader))
imagenes_test = imagenes_test.to('cuda')
 
with torch.no_grad():
    salidas = modelo_gpu(imagenes_test)
    _, predicciones = torch.max(salidas, 1)
 
# Traer todo de vuelta a CPU para visualizar
imagenes_test = imagenes_test.cpu()
predicciones  = predicciones.cpu()
 
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(imagenes_test[i].squeeze(), cmap='gray')
    correcto = predicciones[i] == etiquetas_test[i]
    color = 'green' if correcto else 'red'
    ax.set_title(f'Pred: {predicciones[i]} | Real: {etiquetas_test[i]}',
                 color=color, fontsize=10)
    ax.axis('off')
plt.suptitle('Predicciones del modelo (verde=correcto, rojo=error)', fontsize=12)
plt.tight_layout()
plt.show()"""))

# Section 7
cells.append(nbf.v4.new_markdown_cell("## 7. Prueba tu Propio Dígito"))
cells.append(nbf.v4.new_code_cell("""# Celda 15: Probar propio digito
from google.colab import files
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def procesar_imagen(nombre):
    original = Image.open(nombre).convert('L')
    
    # 1. Recortar bordes (quita sombras y bordes de hoja)
    w, h = original.size
    recortada = original.crop((w*0.05, h*0.05, w*0.95, h*0.95))
    
    # 2. Aumentar contraste para separar trazo del fondo
    from PIL import ImageEnhance, ImageFilter
    contraste = ImageEnhance.Contrast(recortada).enhance(3.0)
    
    # 3. Suavizar ruido de arrugas
    suavizada = contraste.filter(ImageFilter.MedianFilter(size=3))
    
    # 4. Invertir colores (fondo negro, trazo blanco como MNIST)
    invertida = ImageOps.invert(suavizada)
    engrosada = invertida.filter(ImageFilter.MaxFilter(size=3))
    
    # 5. Escalar a 28x28
    procesada = engrosada.resize((28, 28), Image.LANCZOS)
    
    # Visualizar las etapas
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    axes[0].imshow(recortada, cmap='gray')
    axes[0].set_title('1. Recortada')
    axes[0].axis('off')
    
    axes[1].imshow(contraste, cmap='gray')
    axes[1].set_title('2. Contraste')
    axes[1].axis('off')
    
    axes[2].imshow(invertida, cmap='gray')
    axes[2].set_title('3. Invertida')
    axes[2].axis('off')
    
    axes[3].imshow(np.array(procesada), cmap='gray')
    axes[3].set_title('4. Lo que ve la red (28x28)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return procesada

subido = files.upload()
if len(subido) > 0:
    nombre = list(subido.keys())[0]

    imagen = procesar_imagen(nombre)

    # Pasar al modelo
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(imagen).unsqueeze(0).to('cuda')

    modelo_gpu.eval()
    with torch.no_grad():
        salida = modelo_gpu(tensor)
        prediccion = salida.argmax(dim=1).item()

    print(f'El modelo GPU predice: {prediccion}')

    modelo_cpu.eval()
    with torch.no_grad():
        tensor_cpu = transform(imagen).unsqueeze(0).to('cpu')
        salida = modelo_cpu(tensor_cpu)
        prediccion = salida.argmax(dim=1).item()

    print(f'El modelo CPU predice: {prediccion}')
else:
    print('Por favor sube una imagen.')"""))

cells.append(nbf.v4.new_markdown_cell("### Bonus: ¿Qué tan seguro está el modelo?"))
cells.append(nbf.v4.new_code_cell("""# Celda 16: Ver qué tan seguro está cada modelo
import torch.nn.functional as F

if 'imagen' in locals():
    with torch.no_grad():
        # GPU
        tensor_gpu = transform(imagen).unsqueeze(0).to('cuda')
        salida_gpu = modelo_gpu(tensor_gpu)
        prob_gpu = F.softmax(salida_gpu, dim=1)[0]
        
        # CPU
        tensor_cpu = transform(imagen).unsqueeze(0).to('cpu')
        salida_cpu = modelo_cpu(tensor_cpu)
        prob_cpu = F.softmax(salida_cpu, dim=1)[0]

    print("Probabilidades GPU:")
    for i, p in enumerate(prob_gpu):
        print(f"  {i}: {p.item()*100:.1f}%")

    print("\\nProbabilidades CPU:")
    for i, p in enumerate(prob_cpu):
        print(f"  {i}: {p.item()*100:.1f}%")
else:
    print('Debes cargar y procesar una imagen en la celda anterior primero.')"""))

nb.cells = cells
with open('Taller_CUDA_PyTorch.ipynb', 'w') as f:
    nbf.write(nb, f)
