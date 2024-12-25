import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource
from scipy.stats import qmc


class MyPINN1D(tf.keras.Model):
    """
    Физически-информированная нейронная сеть (PINN) для решения краевых задач в одномерной области.

    Этот класс наследует `tf.keras.Model` и реализует PINN для решения дифференциального уравнения вида:
        -(k(x) * u'(x))' + q(x) * u(x) = f(x),
    с краевыми условиями Дирихле:
        u(a) = u_a, u(b) = u_b.

    Атрибуты:
        u_a (float): Значение граничного условия в точке x = a.
        u_b (float): Значение граничного условия в точке x = b.
        k (callable): Функция коэффициента k(x) в дифференциальном уравнении.
        q (callable): Функция коэффициента q(x) в дифференциальном уравнении.
        f (callable): Правая часть f(x) в дифференциальном уравнении.
        x_bord (tuple): Границы области определения уравнения (a, b).
        a (tf.Tensor): Тензорное представление нижней границы (a).
        b (tf.Tensor): Тензорное представление верхней границы (b).
        dense_1 (tf.keras.layers.Dense): Первый полносвязный слой с активацией Swish.
        drop_1 (tf.keras.layers.Dropout): Dropout-слой после `dense_1`.
        dense_2 (tf.keras.layers.Dense): Второй полносвязный слой с активацией Swish.
        drop_2 (tf.keras.layers.Dropout): Dropout-слой после `dense_2`.
        dense_3 (tf.keras.layers.Dense): Третий полносвязный слой с активацией Swish.
        drop_3 (tf.keras.layers.Dropout): Dropout-слой после `dense_3`.
        dense_4 (tf.keras.layers.Dense): Выходной полносвязный слой с линейной активацией.

    Методы:
        call(x):
            Выполняет прямой проход через сеть.
        compile(optimizer):
            Компилирует модель с заданным оптимизатором.
        generate_train_data(data_size):
            Генерирует случайные тренировочные данные в пределах области [a, b].
        residual_loss(x):
            Вычисляет невязку для дифференциального уравнения.
        initial_loss(x):
            Вычисляет начальную потерю (пока не реализовано, возвращает 0).
        boundary_loss():
            Вычисляет потерю, связанную с граничными условиями.
        train_step_v1(batch_size):
            Выполняет один шаг обучения.
        train_v1(epochs, batch_size, interval_info):
            Выполняет обучение модели на заданное число эпох.
    """

    def __init__(self, u_a, u_b, k, q, f, x_bord):
        """
        Инициализирует объект класса MyPINN1D.

        Параметры:
            u_a (float): Значение граничного условия на левом краю (x = a).
            u_b (float): Значение граничного условия на правом краю (x = b).
            k (callable): Функция коэффициента k(x) в дифференциальном уравнении.
            q (callable): Функция коэффициента q(x) в дифференциальном уравнении.
            f (callable): Правая часть уравнения f(x).
            x_bord (tuple): Границы области (a, b), где a и b - крайние точки диапазона.

        Атрибуты:
            u_a (float): Значение граничного условия в точке x = a.
            u_b (float): Значение граничного условия в точке x = b.
            k (callable): Коэффициентная функция k(x).
            q (callable): Коэффициентная функция q(x).
            f (callable): Правая часть f(x).
            x_bord (tuple): Границы области (a, b).
            a (tf.Tensor): Тензорная форма нижней границы (a).
            b (tf.Tensor): Тензорная форма верхней границы (b).
            dense_1 (tf.keras.layers.Dense): Первый полносвязный слой с 50 нейронами и активацией Swish.
            drop_1 (tf.keras.layers.Dropout): Dropout-слой (0.1) после первого слоя.
            dense_2 (tf.keras.layers.Dense): Второй полносвязный слой с 50 нейронами и активацией Swish.
            drop_2 (tf.keras.layers.Dropout): Dropout-слой (0.1) после второго слоя.
            dense_3 (tf.keras.layers.Dense): Третий полносвязный слой с 50 нейронами и активацией Swish.
            drop_3 (tf.keras.layers.Dropout): Dropout-слой (0.1) после третьего слоя.
            dense_4 (tf.keras.layers.Dense): Выходной полносвязный слой с 1 нейроном и линейной активацией.

        """
        super(MyPINN1D, self).__init__()
        self.u_a = u_a
        self.u_b = u_b
        self.k = k
        self.q = q
        self.f = f
        self.x_bord = x_bord
        self.a = tf.reshape(tf.convert_to_tensor([x_bord[0]]), [1, 1])
        self.b = tf.reshape(tf.convert_to_tensor([x_bord[1]]), [1, 1])
        self.dense_1 = tf.keras.layers.Dense(50, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_1 = tf.keras.layers.Dropout(0.1)
        self.dense_2 = tf.keras.layers.Dense(50, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_2 = tf.keras.layers.Dropout(0.1)
        self.dense_3 = tf.keras.layers.Dense(50, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_3 = tf.keras.layers.Dropout(0.1)
        self.dense_4 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, x):
        """
        Выполняет прямой проход через нейронную сеть.

        Параметры:
            x (tf.Tensor): Входной тензор с координатами (размерность [batch_size, 1]).

        Возвращает:
            tf.Tensor: Выходной тензор, представляющий значение функции u(x),
            рассчитанное моделью.
        """
        x = self.dense_1(x)
        x = self.drop_1(x)
        x = self.dense_2(x)
        x = self.drop_2(x)
        x = self.dense_3(x)
        x = self.drop_3(x)
        x = self.dense_4(x)
        return x

    def compile(self, optimizer):
        """
        Компилирует модель с указанным оптимизатором.

        Параметры:
            optimizer (tf.keras.optimizers.Optimizer): Оптимизатор, используемый для обучения модели.
        """
        super(MyPINN1D, self).compile()
        self.optimizer = optimizer

    def generate_train_data(self, data_size):
        """
        Генерирует случайные тренировочные данные в пределах области определения.

        Параметры:
            data_size (int): Количество случайных точек, генерируемых в пределах области [a, b].

        Возвращает:
            tf.Tensor: Тензор формы [data_size, 1], содержащий случайные значения из интервала [a, b].
        """
        x = tf.random.uniform([data_size, 1], self.x_bord[0], self.x_bord[1])
        return x

    @tf.function
    def residual_loss(self, x):
        """
        Вычисляет невязку (residual loss) для дифференциального уравнения.

        Невязка вычисляется как среднеквадратичное отклонение между левой и правой
        частями дифференциального уравнения:
            -(k(x) * u'(x))' + q(x) * u(x) - f(x) = 0.

        Параметры:
            x (tf.Tensor): Тензор с координатами точек, в которых вычисляется невязка
            (размерность [batch_size, 1]).

        Возвращает:
            tf.Tensor: Скалярное значение невязки (среднеквадратичная ошибка).
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.call(x)

            u_x = tape.gradient(u, x)
            ku_x = self.k(x) * u_x
            ku_x_x = tape.gradient(ku_x, x)
            del tape
        residual_loss = tf.reduce_mean(tf.square(-ku_x_x + self.q(x) * u - self.f(x)))
        return residual_loss

    @tf.function
    def initial_loss(self, x):
        """
        Вычисляет начальную потерю (initial loss).

        В текущей реализации возвращает нулевое значение, так как данный метод
        не используется или не реализован.

        Параметры:
            x (tf.Tensor): Входной тензор с координатами (размерность [batch_size, 1]).

        Возвращает:
            int: Нулевое значение (0).
        """
        return 0

    @tf.function
    def boundary_loss(self):
        """
        Вычисляет потерю, связанную с граничными условиями.

        Потеря (boundary loss) рассчитывается как среднеквадратичное отклонение между
        предсказанными значениями u(a) и u(b) и заданными граничными условиями u_a и u_b:
            boundary_loss = mean((u(a) - u_a)^2 + (u(b) - u_b)^2).

        Возвращает:
            tf.Tensor: Скалярное значение потери, связанной с граничными условиями.
        """
        boundary_loss = tf.reduce_mean(tf.square(self.call(self.a) - self.u_a) \
                                       + tf.square(self.call(self.b) - self.u_b))
        return boundary_loss

    @tf.function
    def train_step_v1(self, batch_size):
        """
        Выполняет один шаг обучения модели, включая вычисление потерь и обновление весов.

        В процессе вычисляются две основные потери: невязка (residual_loss) для дифференциального уравнения
        и потеря, связанная с граничными условиями (boundary_loss). Затем, на основе суммарной потери,
        вычисляются градиенты и обновляются веса модели.

        Параметры:
            batch_size (int): Размер партии (batch size) для обучения в данном шаге.

        Возвращает:
            dict: Словарь с двумя значениями:
                - 'residual_loss' (tf.Tensor): Потеря, связанная с невязкой дифференциального уравнения.
                - 'boundary_loss' (tf.Tensor): Потеря, связанная с граничными условиями.
        """
        x = self.generate_train_data(batch_size)

        with tf.GradientTape() as tape:
            residual_loss = self.residual_loss(x)
            boundary_loss = self.boundary_loss()

            total_loss = residual_loss + boundary_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {'residual_loss': residual_loss, 'boundary_loss': boundary_loss}

    def train_v1(self, epochs, batch_size, interval_info=100):
        """
        Обучает модель в течение заданного числа эпох с использованием мини-батчей.

        Для каждой эпохи выполняется один шаг обучения с заданным размером мини-батча.
        Результаты вычислений потерь выводятся на экран каждые `interval_info` эпох.

        Параметры:
            epochs (int): Количество эпох для обучения.
            batch_size (int): Размер мини-батча для каждого шага обучения.
            interval_info (int, по умолчанию 100): Интервал между выводами информации об обучении (эпохи, на которых выводятся потери).

        Вывод:
            На экране выводятся значения потерь (residual_loss и boundary_loss) каждые `interval_info` эпох.
        """
        for epoch in range(epochs):
            res = self.train_step_v1(batch_size)
            if epoch % interval_info == 0:
                print(f"epoch{epoch}, residual_loss  = {res['residual_loss']}, boundary_loss = {res['boundary_loss']}")


class MyPINN2D(tf.keras.Model):
    """
    Класс MyPINN2D реализует PINN (Physics-Informed Neural Network) для решения двумерных задач
    с использованием заданных начальных и граничных условий.

    Параметры:
        g1, g2 : функции
            Граничные условия на границах области.
        v1, v2 : функции
            Начальные условия для значений функции и её производной.
        f : функция
            Правая часть уравнения.
        x_bord : список
            Границы по пространственной координате [x_min, x_max].
        t_bord : список
            Границы по временной координате [t_min, t_max].
    """

    def __init__(self, g1, g2, v1, v2, f, x_bord, t_bord):
        """
        Инициализатор класса MyPINN2D, задающий параметры задачи и архитектуру нейронной сети.

        Параметры:
            g1, g2 : функции
                Граничные условия на границах области.
            v1, v2 : функции
                Начальные условия для значений функции и её производной.
            f : функция
                Правая часть уравнения, описывающая внешний источник или взаимодействие.
            x_bord : список из двух элементов
                Границы по пространственной координате, формат [x_min, x_max].
            t_bord : список из двух элементов
                Границы по временной координате, формат [t_min, t_max].

        Архитектура сети:
            - 4 скрытых слоя, каждый из которых содержит 100 нейронов, функцию активации Swish и инициализацию весов методом Глорота.
            - После каждого скрытого слоя используется слой Dropout с вероятностью 0.1 для предотвращения переобучения.
            - Финальный слой Dense возвращает одно скалярное значение (выход сети).
        """
        super(MyPINN2D, self).__init__()
        self.v1 = v1
        self.v2 = v2
        self.g1 = g1
        self.g2 = g2
        self.f = f
        self.x_bord = x_bord
        self.t_bord = t_bord
        self.dense_1 = tf.keras.layers.Dense(100, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_1 = tf.keras.layers.Dropout(0.1)
        self.dense_2 = tf.keras.layers.Dense(100, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_2 = tf.keras.layers.Dropout(0.1)
        self.dense_3 = tf.keras.layers.Dense(100, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_3 = tf.keras.layers.Dropout(0.1)
        self.dense_4 = tf.keras.layers.Dense(100, activation=tf.keras.activations.swish,
                                             kernel_initializer=tf.keras.initializers.glorot_normal)
        self.drop_4 = tf.keras.layers.Dropout(0.1)
        self.dense_5 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, x, t):
        """
        Выполняет прямой проход через нейронную сеть, принимая входные координаты x и t.

        Параметры:
            x (tf.Tensor): Входной тензор с координатами пространственной переменной (размерность [batch_size, 1]).
            t (tf.Tensor): Входной тензор с координатами временной переменной (размерность [batch_size, 1]).

        Возвращает:
            tf.Tensor: Выходной тензор, представляющий значение функции (например, уставшее или температурное поле),
            рассчитанное моделью с учетом как пространственных, так и временных координат.
        """
        x = tf.stack([x, t], axis=-1)
        x = self.dense_1(x)
        x = self.drop_1(x)
        x = self.dense_2(x)
        x = self.drop_2(x)
        x = self.dense_3(x)
        x = self.drop_3(x)
        x = self.dense_4(x)
        x = self.drop_4(x)
        x = self.dense_5(x)
        return x

    def compile(self, optimizer):
        """
        Компилирует модель с указанным оптимизатором.

        Параметры:
            optimizer (tf.keras.optimizers.Optimizer): Оптимизатор, используемый для обучения модели.
        """
        super(MyPINN2D, self).compile()
        self.optimizer = optimizer

    def generate_train_data(self, data_size):
        """
        Генерирует случайные тренировочные данные для задачи с граничными и начальными условиями.

        Этот метод генерирует случайные значения для:
        - Сетку внутренней области (x_rc, t_rc),
        - Граничные условия для x (x_bc_a, x_bc_b) и t (t_bc_a, t_bc_b),
        - Начальные условия для x (x_ic) и t (t_ic).

        Параметры:
            data_size (int): Количество случайных точек, генерируемых для каждой области (внутренняя, граница, начальная).

        Возвращает:
            tuple: Кортеж, содержащий следующие тензоры:
                - x_rc (tf.Tensor): Тензор с координатами для внутренней области по оси x (размерность [data_size]).
                - t_rc (tf.Tensor): Тензор с координатами для внутренней области по оси t (размерность [data_size]).
                - x_bc_a (tf.Tensor): Тензор с координатами для граничных точек по оси x для x = x_bord[0] (размерность [data_size]).
                - x_bc_b (tf.Tensor): Тензор с координатами для граничных точек по оси x для x = x_bord[1] (размерность [data_size]).
                - t_bc_a (tf.Tensor): Тензор с координатами для граничных точек по оси t для t = t_bord[0] (размерность [data_size]).
                - t_bc_b (tf.Tensor): Тензор с координатами для граничных точек по оси t для t = t_bord[1] (размерность [data_size]).
                - x_ic (tf.Tensor): Тензор с координатами для начальных условий по оси x (размерность [data_size]).
                - t_ic (tf.Tensor): Тензор с координатами для начальных условий по оси t для t = t_bord[0] (размерность [data_size]).
        """
        x_rc = tf.random.uniform([data_size], self.x_bord[0], self.x_bord[1])
        t_rc = tf.random.uniform([data_size], self.t_bord[0], self.t_bord[1])

        x_bc_a = tf.ones_like(t_rc) * self.x_bord[0]
        x_bc_b = tf.ones_like(t_rc) * self.x_bord[1]
        t_bc_a = tf.random.uniform([data_size], self.t_bord[0], self.t_bord[1])
        t_bc_b = tf.random.uniform([data_size], self.t_bord[0], self.t_bord[1])

        x_ic = tf.random.uniform([data_size], self.x_bord[0], self.x_bord[1])
        t_ic = tf.ones_like(t_rc) * self.t_bord[0]
        return x_rc, t_rc, x_bc_a, x_bc_b, t_bc_a, t_bc_b, x_ic, t_ic

    @tf.function
    def residual_loss(self, x, t):
        """
        Вычисляет потерю, связанную с невязкой дифференциального уравнения для задачи.

        В этом методе вычисляется вторичная производная функции по пространственной переменной
        (x) и временной переменной (t) для модели, и оценивается остаточная ошибка дифференциального уравнения
        с учетом исходной функции f(x,t).

        Параметры:
            x (tf.Tensor): Тензор с пространственными координатами (размерность [batch_size, 1]).
            t (tf.Tensor): Тензор с временными координатами (размерность [batch_size, 1]).

        Возвращает:
            tf.Tensor: Потеря, вычисленная как среднеквадратичная ошибка между правой и левой частями дифференциального уравнения.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u = self.call(x, t)
            u_x = tape.gradient(u, x)
            u_t = tape.gradient(u, t)
            u_xx = tape.gradient(u_x, x)
            u_tt = tape.gradient(u_t, x)
            del tape
        residual_loss = tf.reduce_mean(tf.square(u_tt - u_xx - self.f(x, t)))
        return residual_loss

    @tf.function
    def initial_loss(self, x, t):
        """
        Вычисляет потерю для начальных условий задачи.

        Этот метод оценивает отклонение между предсказанными значениями функции u(x,t) и ее производными
        от модели и значениями начальных условий, заданных функциями v1(x) и v2(x).

        Параметры:
            x (tf.Tensor): Тензор с пространственными координатами для начальных условий (размерность [batch_size, 1]).
            t (tf.Tensor): Тензор с временными координатами для начальных условий (размерность [batch_size, 1]).

        Возвращает:
            tf.Tensor: Потеря, вычисленная как среднеквадратичная ошибка между предсказанными и заданными значениями начальных условий.
        """
        with tf.GradientTape() as tape:
            tape.watch(t)
            u = self.call(x, t)
            u_t = tape.gradient(u, t)
        initial_loss = tf.reduce_mean(tf.square(u - self.v1(x)) + tf.square(u_t - self.v2(x)))
        return initial_loss

    @tf.function
    def boundary_loss(self, x1, t1, x2, t2):
        """
        Вычисляет потерю для граничных условий задачи.

        В этом методе оценивается отклонение между предсказанными значениями функции u(x,t) и ее значениями на границе,
        заданными функциями g1(x,t) и g2(x,t).

        Параметры:
            x1 (tf.Tensor): Тензор с пространственными координатами для первой граничной точки (размерность [batch_size, 1]).
            t1 (tf.Tensor): Тензор с временными координатами для первой граничной точки (размерность [batch_size, 1]).
            x2 (tf.Tensor): Тензор с пространственными координатами для второй граничной точки (размерность [batch_size, 1]).
            t2 (tf.Tensor): Тензор с временными координатами для второй граничной точки (размерность [batch_size, 1]).

        Возвращает:
            tf.Tensor: Потеря, вычисленная как среднеквадратичная ошибка между предсказанными и заданными значениями граничных условий.
        """
        boundary_loss = tf.reduce_mean(tf.square(self.call(x1, t1) - self.g1(x1, t1)) \
                                       + tf.square(self.call(x2, t2) - self.g2(x2, t2)))
        return boundary_loss

    @tf.function
    def train_step_v1(self, batch_size):
        """
        Выполняет один шаг обучения модели, включая вычисление потерь и обновление весов.

        В этом методе генерируются тренировочные данные, затем вычисляются потери для невязки уравнения,
        граничных и начальных условий, и обновляются веса модели с использованием градиентного спуска.

        Параметры:
            batch_size (int): Размер пакета для генерации тренировочных данных.

        Возвращает:
            dict: Словарь с тремя ключами:
                - 'residual_loss' (tf.Tensor): Потеря для невязки уравнения.
                - 'boundary_loss' (tf.Tensor): Потеря для граничных условий.
                - 'initial_loss' (tf.Tensor): Потеря для начальных условий.
        """
        x_rc, t_rc, x_bc_a, x_bc_b, t_bc_a, t_bc_b, x_ic, t_ic = self.generate_train_data(batch_size)

        with tf.GradientTape() as tape:
            residual_loss = self.residual_loss(x_rc, t_rc)
            boundary_loss = self.boundary_loss(x_bc_a, t_bc_a, x_bc_b, t_bc_b)
            initial_loss = self.initial_loss(x_ic, t_ic)

            total_loss = residual_loss + boundary_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {'residual_loss': residual_loss, 'boundary_loss': boundary_loss, 'initial_loss': initial_loss}

    def train_v1(self, epochs, batch_size, interval_info=100):
        """
        Обучает модель в течение заданного числа эпох.

        Метод выполняет обучение модели на основе шага обучения `train_step_v1`,
        выводя информацию о потере каждые `interval_info` эпох.

        Параметры:
            epochs (int): Количество эпох для обучения.
            batch_size (int): Размер пакета для генерации тренировочных данных.
            interval_info (int, optional): Частота вывода информации о процессе обучения (по умолчанию 100).

        Выводит:
            str: Каждые `interval_info` эпох выводится информация о значениях потерь для невязки,
                 граничных и начальных условий.
        """
        for epoch in range(epochs):
            res = self.train_step_v1(batch_size)
            if epoch % interval_info == 0:
                print(
                    f"epoch{epoch}, residual_loss  = {res['residual_loss']}, boundary_loss = {res['boundary_loss']}, initial_loss = {res['initial_loss']}")


class StringVibrationAnimation:
    def __init__(self, f, x_bord, t_bord, model_name):
        self.f = f
        self.x_bord = x_bord
        self.t_bord = t_bord
        self.model_name = model_name
    def plot_state(self,t_val,k):
        fig,ax = plt.subplots(figsize=(10,10))
        fig.suptitle(self.model_name)
        t = t_val*np.ones(k)
        x = np.linspace(self.x_bord[0],self.x_bord[1],k)
        res = self.f(x,t)
        ax.scatter(x,res)
    def get_animation(self,k,frame_cnt,interval):
        fig,ax = plt.subplots(figsize=(10,10))
        fig.suptitle(self.model_name)
        frames = []
        t_vals = np.linspace(self.t_bord[0],self.t_bord[1],frame_cnt)
        x = np.linspace(self.x_bord[0],self.x_bord[1],k)
        res = np.zeros((frame_cnt,k))
        for i in range(frame_cnt):
            res[i] = self.f(x,t_vals[i]*np.ones(x.shape)).numpy().flatten()
        ax.set_ylim(res.min(),res.max())
        for i in range(frame_cnt):
            scatter = ax.scatter(x,res[i])
            frames.append([scatter])

        animation = ArtistAnimation(fig,frames,interval=interval,blit=False)
        animation.save(self.model_name+'.gif', writer='pillow')
    def get_compare_animation(self,true_func,k,frame_cnt,interval):
        fig,ax = plt.subplots(1,2,figsize=(10,10))
        fig.suptitle('Сравнение ' + self.model_name + ' с реальной зависимостью')
        frames = []
        t_vals = np.linspace(self.t_bord[0],self.t_bord[1],frame_cnt)
        x = np.linspace(self.x_bord[0],self.x_bord[1],k)
        res = np.zeros((frame_cnt,k))
        tr_val = np.zeros((frame_cnt,k))
        for i in range(frame_cnt):
            res[i] = self.f(x,t_vals[i]*np.ones(x.shape)).numpy().flatten()
            tr_val[i] = true_func(x, t_vals[i]*np.ones(x.shape))
        ax[0].set_ylim(res.min(),res.max())
        for i in range(frame_cnt):
            scatter = ax[0].scatter(x,res[i])
            scatter_pogr = ax[1].scatter(x,np.abs(res[i]-tr_val[i]))
            frames.append([scatter,scatter_pogr])

        animation = ArtistAnimation(fig,frames,interval=interval,blit=False)
        animation.save('comparation'+self.model_name+'.gif', writer='pillow')