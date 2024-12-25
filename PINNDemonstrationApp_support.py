#! /usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 8.0
#  in conjunction with Tcl version 8.6
#    Dec 25, 2024 03:00:07 AM +03  platform: Windows NT

import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
import threading

from sympy import false
import time

from MyPINN import MyPINN1D,MyPINN2D,tf,plt,np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import PINNDemonstrationApp

_debug = True # False to eliminate debug printing from callback functions.

def main(*args):
    '''Main entry point for the application.'''
    global root
    root = tk.Tk()
    root.protocol( 'WM_DELETE_WINDOW' , root.destroy)
    # Creates a toplevel widget.
    global _top2, _w2
    _top2 = root
    _w2 = PINNDemonstrationApp.mainWnd(_top2)
    # Creates a toplevel widget.
    global _top3, _w3
    _top3 = tk.Toplevel(root)
    _w3 = PINNDemonstrationApp.Problem1(_top3)
    _top3.protocol("WM_DELETE_WINDOW", prevent_closing_1)
    _top3.withdraw()
    # Creates a toplevel widget.
    global _top4, _w4
    _top4 = tk.Toplevel(root)
    _w4 = PINNDemonstrationApp.Problem2(_top4)
    _top4.protocol("WM_DELETE_WINDOW", prevent_closing_2)
    _top4.withdraw()
    root.mainloop()


    global epochs1, epochs2, batch_size1,batch_size2,info_interval1, info_interval2
    global pinn1D, pinn2D,created1,created2,train2
    global a_1, b_1, u_a_1, u_b_1, res_loss_1_arr, boundary_loss_1_arr, loss_1_arr
    global a_2, b_2, c_2, d_2, init_loss_2_arr, boundary_loss_2_arr, loss_2_arr, res_loss_2_arr
    global animation_thread
    animation_thread = None

    res_loss_1_arr, boundary_loss_1_arr, loss_1_arr = [],[],[]
    res_loss_2_arr, boundary_loss_2_arr, loss_2_arr, init_loss_2_arr = [], [], [], []
    created1 = False
    created2 = False
    global stop_flag
    stop_flag = False

def batch_size_1_cnhg(*args):
    _w3.batch_size_1_info.configure(text=f'{int(_w3.batch_size_1_scale.get())}')
    if _debug:
        print('PINNDemonstrationApp_support.batch_size_1_cnhg')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def batch_size_2_chng(*args):
    _w4.batch_size_2_info.configure(text=f'{int(_w4.batch_size_2_scale.get())}')
    if _debug:
        print('PINNDemonstrationApp_support.batch_size_2_chng')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def create_model_1(*args):
    global created1,pinn1D,a_1,b_1,u_a_1,u_b_1
    try:
        f1 = eval('lambda x:' + _w3.f_1_input.get())
        q1 = eval('lambda x:' + _w3.q_1_input.get())
        k1 = eval('lambda x:' + _w3.k_1_input.get())
        a_1 = float(_w3.a_1_input.get())
        b_1 = float(_w3.b_1_input.get())
        u_a_1 = float(_w3.u_a_1_input.get())
        u_b_1 = float(_w3.u_b_1_input.get())
        pinn1D = MyPINN1D(u_a_1,u_b_1,k1,q1,f1,tf.convert_to_tensor([a_1,b_1]))
        _w3.list_report_1.delete(0,tk.END)
        _w3.list_report_1.insert(tk.END,f'Модель создана с параметрами: a = {a_1}, b = {b_1}, u(a) = {u_a_1}, u(b) ={u_b_1}')
        created1 = True
    except Exception as e:
        _w3.list_report_1.delete(0, tk.END)
        _w3.list_report_1.insert(tk.END,
                                 f'Произошла ошибка при построении модели: {e}')


    if _debug:
        print('PINNDemonstrationApp_support.create_model_1')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def create_model_2(*args):
    global created2, pinn2D, a_2, b_2, c_2, d_2
    try:
        f2 = eval('lambda x,t:' + _w4.f_2_input.get())
        g2 = eval('lambda x,t:' + _w4.g_2_input.get())
        g22 = eval('lambda x,t:' + _w4.g_22_input.get())
        v1 = eval('lambda x:' + _w4.v_1_input.get())
        v2 = eval('lambda x:' + _w4.v_2_input.get())
        a_2 = float(_w4.a_2_input.get())
        b_2 = float(_w4.b_2_input.get())
        c_2 = float(_w4.c_2_input.get())
        d_2 = float(_w4.d_2_input.get())
        pinn2D = MyPINN2D(g2,g22,v1,v2,f2,tf.convert_to_tensor([a_2,b_2]),tf.convert_to_tensor([c_2,d_2]))
        _w4.list_report_2.delete(0, tk.END)
        _w4.list_report_2.insert(tk.END,
                                 f'Модель создана с параметрами: a = {a_2}, b = {b_2}, c = {c_2}, d = {d_2}')
        created2 = True
    except Exception as e:
        _w4.list_report_2.delete(0, tk.END)
        _w4.list_report_2.insert(tk.END,
                                 f'Произошла ошибка при построении модели: {e}')
    if _debug:
        print('PINNDemonstrationApp_support.create_model_2')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def epochs_1_chng(*args):
    _w3.epochs_1_info.configure(text=f'{int(_w3.epochs_1_scale.get())}')
    if _debug:
        print('PINNDemonstrationApp_support.epochs_1_chng')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def epochs_2_chng(*args):
    _w4.epochs_2_info.configure(text=f'{int(_w4.epochs_2_scale.get())}')
    if _debug:
        print('PINNDemonstrationApp_support.epochs_2_chng')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def exit_event(*args):
    root.destroy()
    if _debug:
        print('PINNDemonstrationApp_support.exit_event')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def go_to_problem1(*args):
    _top3.deiconify()
    if _debug:
        print('PINNDemonstrationApp_support.go_to_problem1')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def go_to_problem2(*args):
    _top4.deiconify()
    if _debug:
        print('PINNDemonstrationApp_support.go_to_problem2')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def interval_1_chng(*args):
    _w3.interval_1_info.configure(text=f'{int(_w3.info_int_1_scale.get())}')
    if _debug:
        print('PINNDemonstrationApp_support.interval_1_chng')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def interval_2_chng(*args):
    _w4.interval_2_info.configure(text=f'{int(_w4.info_int_2_scale.get())}')
    if _debug:
        print('PINNDemonstrationApp_support.interval_2_chng')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def train_model_1(*args):
    global created1 ,pinn1D,res_loss_1_arr, boundary_loss_1_arr, loss_1_arr
    if created1:
        try:
            res_loss_1_arr, boundary_loss_1_arr, loss_1_arr = [], [], []
            optimizer = tf.keras.optimizers.Adam(1e-3)
            pinn1D.compile(optimizer=optimizer)
            _w3.list_report_1.insert(tk.END, f'Модель успешно скомпилирована. Сейчас начнется тренировка с параметрами: \
                        epochs = {int(_w3.epochs_1_scale.get())}, batch_size = {int(_w3.batch_size_1_scale.get())},\
                        info interval = {int(_w3.info_int_1_scale.get())}')
            train1 = threading.Thread(target=pinn1D.train_v1, args= (int(_w3.epochs_1_scale.get()),int(_w3.batch_size_1_scale.get()),int(_w3.info_int_1_scale.get()),callback_func_1))
            res_loss_1_arr, boundary_loss_1_arr, loss_1_arr = [], [], []
            train1.start()
            def  wait_to_draw():
                train1.join()
                draw_graph_1()
            draw_1 = threading.Thread(target=wait_to_draw)
            draw_1.start()
        except Exception as e:
            _w3.list_report_1.insert(tk.END,f'Произошла ошибка:{e}')
    if _debug:
        print('PINNDemonstrationApp_support.train_model_1')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def train_model_2(*args):
    global created2, pinn2D, res_loss_2_arr, boundary_loss_2_arr, loss_2_arr, init_loss_2_arr, c_2,d_2, stop_flag
    if created2:
        try:
            res_loss_2_arr, boundary_loss_2_arr, loss_2_arr, init_loss_2_arr = [], [], [], []
            try:
                stop_flag = True
            except Exception as e:
                print(f'Ошибка {e}')
            optimizer = tf.keras.optimizers.Adam(1e-3)
            pinn2D.compile(optimizer=optimizer)
            _w4.list_report_2.insert(tk.END, f'Модель успешно скомпилирована. Сейчас начнется тренировка с параметрами: \
            epochs = {int(_w4.epochs_2_scale.get())}, batch_size = {int(_w4.batch_size_2_scale.get())},\
            info interval = {int(_w4.info_int_2_scale.get())}')
            train2 = threading.Thread(target=pinn2D.train_v1, args=(
            int(_w4.epochs_2_scale.get()), int(_w4.batch_size_2_scale.get()), int(_w4.info_int_2_scale.get()),
            callback_func_2))
            train2.start()

            k = 100
            t_vals = np.linspace(c_2,d_2,k)
            def func_to_ani():
                train2.join()
                draw_graph_2(t_vals)
            stop_flag = False
            draw_2 = threading.Thread(target=func_to_ani)
            draw_2.start()
        except Exception as e:
            _w4.list_report_2.insert(tk.END, f'Произошла ошибка:{e}')
    if _debug:
        print('PINNDemonstrationApp_support.train_model_2')
        for arg in args:
            print ('    another arg:', arg)
        sys.stdout.flush()

def prevent_closing_1():
    _top3.withdraw()

def prevent_closing_2():
    global stop_flag
    stop_flag = True
    _top4.withdraw()

def callback_func_1(res,epoch):
    global res_loss_1_arr, boundary_loss_1_arr, loss_1_arr
    _w3.list_report_1.insert(tk.END, f"epoch: {epoch},\
     residual_loss : {res['residual_loss']},\
      boundary_loss: {res['boundary_loss']}")
    res_loss_1_arr.append(res['residual_loss'])
    boundary_loss_1_arr.append(res['boundary_loss'])
    loss_1_arr.append(res['boundary_loss'] + res['residual_loss'])
    print("now i'm here")

def callback_func_2(res,epoch):
    global res_loss_2_arr, boundary_loss_2_arr, loss_2_arr , init_loss_2_arr
    print("now i'm here2")
    res_loss_2_arr.append(res['residual_loss'])
    boundary_loss_2_arr.append(res['boundary_loss'])
    init_loss_2_arr.append(res['initial_loss'])
    loss_2_arr.append(res['boundary_loss'] + res['residual_loss'] + res['initial_loss'])

    _w4.list_report_2.insert(tk.END, f"epoch: {epoch}, residual_loss : {res['residual_loss']},\
      boundary_loss: {res['boundary_loss']}, initial_loss: {res['initial_loss']}")


def draw_graph_1():
    # Создаем объект Figure
    fig,ax  = plt.subplots(1,2,figsize=(10, 5), dpi=100)
    global  a_1, b_1
    k = 100
    # Данные для графика
    x = np.linspace(a_1, b_1, k)
    y = pinn1D(tf.reshape(tf.convert_to_tensor(x),[k,1]))

    # Построение графика
    ax[0].plot(x, y, color='#ff00ff',linestyle='--')
    ax[0].set_title("Предсказание модели")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()
    ax[0].set_xticks(np.linspace(a_1,b_1,11),minor=True)
    ax[0].grid(linestyle='-.',which='minor',alpha=0.5)
    ax[0].grid(alpha=0.75)

    global loss_1_arr, res_loss_1_arr, boundary_loss_1_arr
    k1 = len(loss_1_arr)
    x = np.arange(0,k1,1)

    ax[1].plot(x, np.array(res_loss_1_arr), color = 'red', linestyle='-.', label ='residual_loss')
    ax[1].plot(x, np.array(loss_1_arr), color ='green', label = 'total_loss')
    ax[1].plot(x, np.array(boundary_loss_1_arr), color = 'blue', linestyle='-.',label = 'boundary_loss')
    ax[1].set_title('Динамика обучения')
    ax[1].legend()
    ax[1].grid(linestyle='-.',  alpha=0.775)
    ax[1].set_yscale('log')

    # Вставка Figure в Canvas Tkinter
    canvas = FigureCanvasTkAgg(fig, master=_w3.presentation_frame_1)  # Связываем с Tkinter
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.place(relx=0.01, rely=0.05, relheight=0.45
                , relwidth=0.98)
    canvas.draw()

def draw_graph_2(t_vals):

    global loss_2_arr, res_loss_2_arr, boundary_loss_2_arr, init_loss_2_arr, pinn2D
    global stop_flag,a_2,b_2
    fig,ax  = plt.subplots(1,2,figsize=(10, 5), dpi=100)

    canvas = FigureCanvasTkAgg(fig, master=_w4.presentation_frame_2)  # Связываем с Tkinter
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.place(relx=0.01, rely=0.05, relheight=0.45, relwidth=0.981)

    k1 = len(loss_2_arr)
    x = np.arange(0, k1, 1)

    ax[1].plot(x, np.array(res_loss_2_arr), color='red', linestyle='-.', label='residual_loss')
    ax[1].plot(x, np.array(loss_2_arr), color='green', label='total_loss')
    ax[1].plot(x, np.array(boundary_loss_2_arr), color='blue', linestyle='-.', label='boundary_loss')
    ax[1].set_title('Динамика обучения')
    ax[1].legend()
    ax[1].grid(linestyle='-.', alpha=0.75)
    ax[1].set_yscale('log')

    k = 100
    x = np.linspace(a_2, b_2, k)
    x_ = tf.convert_to_tensor(x)
    y_all = [pinn2D(x_, t_val * tf.ones_like(x_)) for t_val in t_vals]
    y_np = np.array(y_all)


    scatter_plot = None

    ax[0].set_title("Предсказание модели")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_ylim([y_np.min(),y_np.max()])
    ax[0].set_xticks(np.linspace(a_2, b_2, 11), minor=True)
    ax[0].grid(linestyle='-.', which='minor', alpha=0.5)
    ax[0].grid(alpha=0.75)

    while not stop_flag:
        for y in y_all:
            # Данные для графика
            time.sleep(0.05)

            if scatter_plot:
                scatter_plot.remove()

                # Создаем новый scatter
            scatter_plot = ax[0].scatter(x, y, color='#ff00ff')

            # Вставка Figure в Canvas Tkinter
            canvas.draw()

if __name__ == '__main__':
    PINNDemonstrationApp.start_up()




