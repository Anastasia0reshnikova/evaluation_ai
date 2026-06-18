"""
Задание 4.

Выполните глубокое обучение на датасете MNIST, используя подготовленный пример из файла classification.py
и новые архитектуры моделей (VerySimpleCNN и SimpleCNN).

Результатом выполнения задания является текстовый отчет, где отображены:
- результаты оценки моделей;
- полученные данные;
- графики;
- выводы по результатам.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from scipy.stats import entropy as scipy_entropy
from torchmetrics.classification import MulticlassCalibrationError


# ============================================================================
# 1. НАСТРОЙКА УСТРОЙСТВА
# ============================================================================

# Здесь мы выбираем, где будет обучаться модель:
# - если доступна видеокарта NVIDIA с CUDA, используется GPU;
# - иначе используется CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


# ============================================================================
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ MNIST
# ============================================================================

def load_mnist_data(batch_size=128, val_split=0.1):
    """
    Загружает датасет MNIST и подготавливает его для обучения.

    MNIST содержит изображения рукописных цифр:
    - размер изображения: 28x28 пикселей;
    - цветность: grayscale, то есть 1 канал;
    - классы: цифры от 0 до 9;
    - train dataset: 60 000 изображений;
    - test dataset: 10 000 изображений.

    Здесь мы делим данные на:
    - train;
    - validation;
    - test.
    """

    # Трансформации для обучающего набора.
    # Здесь добавлена небольшая аугментация:
    # RandomCrop — немного смещает изображение;
    # RandomRotation — немного поворачивает цифру.
    # Это помогает модели лучше обобщать данные.
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Для validation и test аугментация не нужна.
    # Мы только переводим изображения в Tensor и нормализуем.
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Загружаем полный train dataset MNIST.
    full_train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    # Загружаем test dataset MNIST.
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform_val_test
    )

    # Делим train dataset на train и validation.
    # Например, если val_split=0.1:
    # 90% данных идет на обучение,
    # 10% данных идет на валидацию.
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Фиксируем random seed, чтобы разделение было воспроизводимым.
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset_temp = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Создаем отдельный validation dataset без аугментации.
    # Это важно, потому что validation должен показывать реальное качество модели,
    # а не качество на случайно измененных изображениях.
    val_dataset_full = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=transform_val_test
    )

    # Берем те же индексы, которые попали в validation.
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # DataLoader разбивает данные на batch-и.
    # Это нужно, чтобы модель обучалась не на всех данных сразу, а частями.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Названия классов MNIST.
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    print("Размеры наборов данных:")
    print(f"  - Обучающий: {len(train_dataset)} образцов")
    print(f"  - Валидационный: {len(val_dataset)} образцов")
    print(f"  - Тестовый: {len(test_dataset)} образцов")

    return train_loader, val_loader, test_loader, classes


# ============================================================================
# 3. АРХИТЕКТУРЫ МОДЕЛЕЙ
# ============================================================================

class VerySimpleCNN(nn.Module):
    """
    Очень простая сверхточная нейронная сеть для MNIST.

    Эта модель используется как базовый вариант.
    У нее меньше слоев и меньше параметров, поэтому она быстрее обучается,
    но может быть менее точной, чем более глубокая модель.
    """

    def __init__(self, num_classes=10):
        super(VerySimpleCNN, self).__init__()

        # Первый сверхточный слой.
        # На вход приходит 1 канал, потому что MNIST — grayscale.
        # На выходе получаем 16 feature maps.
        self.conv1 = nn.Conv2d(1, 16, 5)

        # MaxPool уменьшает размер изображения в 2 раза.
        self.pool = nn.MaxPool2d(2, 2)

        # Второй сверхточный слой.
        # На вход приходит 16 каналов, на выходе получаем 32.
        self.conv2 = nn.Conv2d(16, 32, 5)

        # Полно связные слои.
        # После двух conv + pool размер становится 32 * 4 * 4.
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Первый сверхточный блок: conv -> relu -> pooling.
        x = self.pool(torch.relu(self.conv1(x)))

        # Второй сверхточный блок: conv -> relu -> pooling.
        x = self.pool(torch.relu(self.conv2(x)))

        # Превращаем многомерный тензор в одномерный вектор.
        x = x.view(-1, 32 * 4 * 4)

        # Полносвязный слой + ReLU.
        x = torch.relu(self.fc1(x))

        # Выходной слой.
        # Здесь модель возвращает logits для 10 классов.
        x = self.fc2(x)

        return x


class SimpleCNN(nn.Module):
    """
    Более глубокая сверхточная нейронная сеть для MNIST.

    Эта модель сложнее, чем VerySimpleCNN:
    - больше сверхточных слоев;
    - больше нейронов;
    - используется Dropout.

    Ожидается, что такая модель может показать более высокую точность,
    но при этом будет обучаться дольше.
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Сверхточные слои.
        # Входной канал = 1, потому что MNIST grayscale.
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Dropout помогает уменьшить переобучение.
        self.dropout = nn.Dropout(0.5)

        # После трех pooling на изображении 28x28 размер становится 3x3.
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14.
        x = self.pool(self.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 7x7.
        x = self.pool(self.relu(self.conv2(x)))

        # Conv block 3: 7x7 -> 3x3.
        x = self.pool(self.relu(self.conv3(x)))

        # Дополнительный conv layer без pooling.
        x = self.relu(self.conv4(x))

        # Flatten.
        x = x.view(x.size(0), -1)

        # Fully connected layers.
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Финальный слой возвращает logits для 10 классов.
        x = self.fc3(x)

        return x


# ============================================================================
# 4. ВАЛИДАЦИЯ МОДЕЛИ
# ============================================================================

def validate_model(model, val_loader, criterion, device):
    """
    Проверяет качество модели на validation dataset.

    Validation используется во время обучения, чтобы понять,
    улучшается ли модель на данных, которые она не использует для обновления весов.
    """

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() отключает расчет градиентов.
    # Это ускоряет evaluation и экономит память.
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Получаем предсказания модели.
            outputs = model(inputs)

            # Считаем loss.
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Берем класс с максимальным значением logit.
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


# ============================================================================
# 5. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Обучает модель и сохраняет историю обучения.

    На каждой эпохе:
    1. Модель обучается на train dataset.
    2. Модель проверяется на validation dataset.
    3. Сохраняются loss и accuracy.
    """

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Early stopping:
    # если validation accuracy не улучшается несколько эпох подряд,
    # обучение можно остановить раньше.
    best_val_accuracy = 0.0
    patience = 3
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Обнуляем старые градиенты.
            optimizer.zero_grad()

            # Forward pass: модель делает предсказание.
            outputs = model(inputs)

            # Считаем ошибку.
            loss = criterion(outputs, labels)

            # Backward pass: считаем градиенты.
            loss.backward()

            # Обновляем веса модели.
            optimizer.step()

            running_loss += loss.item()

            # Получаем predicted class.
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Метрики на train за эпоху.
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Метрики на validation за эпоху.
        val_loss, val_accuracy = validate_model(
            model,
            val_loader,
            criterion,
            device
        )

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Эпоха [{epoch + 1}/{num_epochs}]:")
        print(f"  Обучение  - Потеря: {train_loss:.4f}, Точность: {train_accuracy:.2f}%")
        print(f"  Валидация - Потеря: {val_loss:.4f}, Точность: {val_accuracy:.2f}%")
        print("-" * 60)

        # Проверяем, улучшилась ли validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            print(f"  ✅ Новая лучшая валидационная точность: {best_val_accuracy:.2f}%")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"  ⏹️ Ранний останов: {patience} эпох без улучшения")
                break

        print()

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_accuracy
    }


# ============================================================================
# 6. ФИНАЛЬНАЯ ОЦЕНКА МОДЕЛИ НА TEST DATASET
# ============================================================================

def evaluate_model_with_metrics(model, test_loader, device):
    """
    Получает предсказания модели на test dataset.

    Test dataset не используется во время обучения.
    Он нужен только для финальной честной оценки модели.
    """

    model.eval()

    all_predictions = []
    all_labels = []
    all_softmax_probs = []
    all_logits = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Logits — сырые выходы модели.
            logits = model(inputs)

            # Softmax превращает logits в вероятности по классам.
            softmax_probs = torch.softmax(logits, dim=1)

            # Предсказанный класс — класс с максимальным logit.
            _, predicted = torch.max(logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_softmax_probs.extend(softmax_probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_softmax_probs),
        np.array(all_logits)
    )


# ============================================================================
# 7. ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ
# ============================================================================

def calculate_confidence_scores(softmax_probs):
    """
    Confidence score — это максимальная вероятность softmax.

    Например, если модель сказала:
    [0.01, 0.02, 0.95, 0.02],
    то confidence score = 0.95.
    """

    return np.max(softmax_probs, axis=1)


def calculate_top_k_accuracy(softmax_probs, true_labels, k=5):
    """
    Top-K Accuracy показывает, попал ли правильный класс
    в k наиболее вероятных предсказаний модели.

    Для MNIST Top-1 Accuracy — это обычная accuracy.
    """

    accuracy = top_k_accuracy_score(true_labels, softmax_probs, k=k)
    return accuracy * 100.0


def calculate_entropy(softmax_probs):
    """
    Энтропия показывает неопределенность модели.

    Низкая энтропия означает, что модель уверена.
    Высокая энтропия означает, что модель сомневается между несколькими классами.
    """

    return scipy_entropy(softmax_probs, axis=1)


def calculate_calibration_error(softmax_probs, true_labels, n_bins=10):
    """
    Expected Calibration Error показывает,
    насколько уверенность модели соответствует реальной точности.

    Если ECE низкий, значит модель хорошо откалибрована.
    """

    probs_t = torch.from_numpy(softmax_probs).float()
    targets_t = torch.from_numpy(true_labels).long()

    ece_metric = MulticlassCalibrationError(
        num_classes=softmax_probs.shape[1],
        n_bins=n_bins,
        norm="l1"
    )

    ece = float(ece_metric(probs_t, targets_t).item())

    return ece


# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_training_history(training_history, model_name):
    """
    Построение графиков accuracy и loss по эпохам.
    """
    train_accuracies = np.asarray(training_history["train_accuracies"], dtype=float)
    val_accuracies = np.asarray(training_history["val_accuracies"], dtype=float)

    train_losses = np.asarray(training_history["train_losses"], dtype=float)
    val_losses = np.asarray(training_history["val_losses"], dtype=float)

    epochs = np.arange(1, len(train_accuracies) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_accuracies, label="Обучение", linewidth=2)
    ax1.plot(epochs, val_accuracies, label="Валидация", linewidth=2)
    ax1.set_title(f"{model_name}: Accuracy по эпохам")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_losses, label="Обучение", linewidth=2)
    ax2.plot(epochs, val_losses, label="Валидация", linewidth=2)
    ax2.set_title(f"{model_name}: Loss по эпохам")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()

    final_train_acc = float(train_accuracies[-1])
    final_val_acc = float(val_accuracies[-1])
    gap = final_train_acc - final_val_acc

    print("\n🔍 АНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
    print(f"Финальная точность обучения: {final_train_acc:.2f}%")
    print(f"Финальная точность валидации: {final_val_acc:.2f}%")
    print(f"Разрыв Train - Val: {gap:.2f}%")

    if gap < 3:
        print("✅ Переобучения практически нет")
    elif gap < 8:
        print("⚠️ Легкое переобучение")
    elif gap < 15:
        print("🔶 Умеренное переобучение")
    else:
        print("❌ Сильное переобучение")

    return gap


def plot_confusion_matrix(true_labels, predictions, classes, model_name):
    """
    Построение confusion matrix.
    """
    cm_abs = confusion_matrix(true_labels, predictions)

    with np.errstate(invalid="ignore"):
        cm = cm_abs / cm_abs.sum(axis=1, keepdims=True)

    cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Доля по истинному классу"}
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j] * 100:.1f}%\n({cm_abs[i, j]})"
            ax.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="black"
            )

    plt.title(f"{model_name}: Матрица ошибок")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================================
# 6. ПАЙПЛАЙН ДЛЯ ОДНОЙ МОДЕЛИ
# ============================================================================

def run_experiment(model_class, model_name, train_loader, val_loader, test_loader, classes):
    """
    Запускает полный эксперимент для одной модели.

    Что происходит внутри:
    1. Создается модель.
    2. Определяются loss function и optimizer.
    3. Модель обучается на train dataset.
    4. Модель проверяется на validation dataset.
    5. После обучения модель оценивается на test dataset.
    6. Считаются метрики.
    7. Строятся графики.
    8. Возвращаются результаты для итогового сравнения.
    """
    print("\n" + "=" * 80)
    print(f"ЗАПУСК ЭКСПЕРИМЕНТА ДЛЯ МОДЕЛИ: {model_name}")
    print("=" * 80)

    model = model_class(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров в модели {model_name}: {total_params:,}")

    print("\nОбучение модели с валидацией...")
    training_history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=10
    )

    print(f"\n🎯 ИТОГИ ОБУЧЕНИЯ {model_name}:")
    print(f"Лучшая валидационная точность: {training_history['best_val_accuracy']:.2f}%")

    print("\nФинальная оценка на тестовом наборе...")
    predictions, true_labels, softmax_probs, logits = evaluate_model_with_metrics(
        model,
        test_loader,
        device
    )

    test_accuracy = accuracy_score(true_labels, predictions) * 100
    val_accuracy = training_history["best_val_accuracy"]

    confidence_scores = calculate_confidence_scores(softmax_probs)
    avg_confidence = np.mean(confidence_scores)

    entropy_values = calculate_entropy(softmax_probs)
    avg_entropy = np.mean(entropy_values)

    ece = calculate_calibration_error(softmax_probs, true_labels)

    top_1_accuracy = calculate_top_k_accuracy(softmax_probs, true_labels, k=1)
    top_3_accuracy = calculate_top_k_accuracy(softmax_probs, true_labels, k=3)
    top_5_accuracy = calculate_top_k_accuracy(softmax_probs, true_labels, k=5)

    print("\n" + "=" * 50)
    print(f"ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТОВОМ НАБОРЕ: {model_name}")
    print("=" * 50)

    print(f"Лучшая валидационная точность: {val_accuracy:.2f}%")
    print(f"Финальная тестовая точность: {test_accuracy:.2f}%")
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top_3_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2f}%")
    print(f"Средний Confidence Score: {avg_confidence:.3f}")
    print(f"Средняя энтропия: {avg_entropy:.3f}")
    print(f"Expected Calibration Error: {ece:.3f}")

    print("\nДетальный отчет по классам:")
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=classes,
        digits=3
    )
    print(class_report)

    print("\nВизуализация результатов...")
    overfitting_gap = plot_training_history(training_history, model_name)
    plot_confusion_matrix(true_labels, predictions, classes, model_name)

    results = {
        "model_name": model_name,
        "total_params": total_params,
        "best_val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "avg_confidence": avg_confidence,
        "avg_entropy": avg_entropy,
        "ece": ece,
        "overfitting_gap": overfitting_gap
    }

    return results


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    """
    Главная функция.

    Здесь выполняется весь pipeline домашнего задания:
    1. Загружается MNIST dataset.
    2. Обучается VerySimpleCNN.
    3. Обучается SimpleCNN.
    4. Сравниваются результаты двух моделей.
    """
    print("=" * 80)
    print("ДОМАШНЕЕ ЗАДАНИЕ 4: MNIST CLASSIFICATION")
    print("=" * 80)

    print("\n1. Загрузка данных MNIST...")
    train_loader, val_loader, test_loader, classes = load_mnist_data(
        batch_size=128,
        val_split=0.1
    )

    print(f"Классы: {classes}")

    results = []

    result_very_simple = run_experiment(
        VerySimpleCNN,
        "VerySimpleCNN",
        train_loader,
        val_loader,
        test_loader,
        classes
    )
    results.append(result_very_simple)

    result_simple = run_experiment(
        SimpleCNN,
        "SimpleCNN",
        train_loader,
        val_loader,
        test_loader,
        classes
    )
    results.append(result_simple)

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)

    for result in results:
        print(f"\nМодель: {result['model_name']}")
        print(f"Количество параметров: {result['total_params']:,}")
        print(f"Лучшая validation accuracy: {result['best_val_accuracy']:.2f}%")
        print(f"Test accuracy: {result['test_accuracy']:.2f}%")
        print(f"Top-3 accuracy: {result['top_3_accuracy']:.2f}%")
        print(f"Top-5 accuracy: {result['top_5_accuracy']:.2f}%")
        print(f"Average confidence: {result['avg_confidence']:.3f}")
        print(f"Average entropy: {result['avg_entropy']:.3f}")
        print(f"ECE: {result['ece']:.3f}")
        print(f"Overfitting gap: {result['overfitting_gap']:.2f}%")

    print("\nАНАЛИЗ MNIST ЗАВЕРШЕН УСПЕШНО!")


# ============================================================================
# 8. ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    main()