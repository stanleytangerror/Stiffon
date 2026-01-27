

import pygame
import jax.numpy as np
from dynamics import CartPole, Pendulum, rk4

def draw_pendulum(states, dt):
    # 初始化 pygame
    pygame.init()

    # 窗口设置
    WIDTH, HEIGHT = 1200, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum Visualization")

    # 颜色定义
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)

    # 物理参数
    BALL_RADIUS = 15
    PENDULUM_LENGTH = 200  # 像素长度，对应 l=0.5m
    PIVOT_X = WIDTH // 2  # 支点在屏幕中心
    PIVOT_Y = HEIGHT // 2
    SCALE = 400  # 将物理单位（米）转换为像素

    # 时间参数
    FPS = 60
    frame_time = 1.0 / FPS  # 每帧的时间间隔（秒）

    # 主循环
    clock = pygame.time.Clock()
    current_time = 0.0  # 当前时间（秒）
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    current_time = 0.0  # 重置

        if not paused:
            current_time += frame_time

        # 计算总时间长度
        total_time = len(states) * dt
        
        # 循环时间
        current_time = current_time % total_time

        # 计算当前时间对应的状态索引（浮点数）
        state_index = current_time / dt
        
        # 获取插值索引
        idx0 = int(state_index) % len(states)
        idx1 = (idx0 + 1) % len(states)
        
        # 计算插值系数
        alpha = state_index - int(state_index)
        
        # 线性插值状态
        state0 = states[idx0]
        state1 = states[idx1]
        state = state0 * (1 - alpha) + state1 * alpha

        # 清屏
        SCREEN.fill(WHITE)

        # 获取当前状态
        theta = state[0]  # 摆的角度（弧度，相对于竖直向下）
        theta_dot = state[1]  # 角速度（弧度/秒）

        # 计算摆的末端位置（theta 是相对于竖直向下的角度）
        # 竖直向下为正方向，theta=0 时摆竖直向下
        ball_x = float(PIVOT_X + PENDULUM_LENGTH * np.sin(theta))
        ball_y = float(PIVOT_Y + PENDULUM_LENGTH * np.cos(theta))

        # 绘制支点
        pygame.draw.circle(SCREEN, BLACK, (PIVOT_X, PIVOT_Y), 8)
        pygame.draw.circle(SCREEN, GRAY, (PIVOT_X, PIVOT_Y), 6)

        # 绘制摆杆（从支点到摆的末端）
        pygame.draw.line(SCREEN, BLACK, (PIVOT_X, PIVOT_Y), (int(ball_x), int(ball_y)), 3)

        # 绘制摆的末端（小球）
        pygame.draw.circle(SCREEN, RED, (int(ball_x), int(ball_y)), BALL_RADIUS)
        pygame.draw.circle(SCREEN, BLACK, (int(ball_x), int(ball_y)), BALL_RADIUS, 2)

        # 显示信息
        font = pygame.font.Font(None, 36)
        info_text = [
            f"Time: {current_time:.3f} s / {total_time:.3f} s",
            f"State Index: {state_index:.2f} ({idx0} -> {idx1}, α={alpha:.2f})",
            f"Theta: {np.degrees(theta):.2f}°",
            f"Theta Dot: {theta_dot:.3f} rad/s",
            "",
            "Press SPACE to pause/resume",
            "Press R to reset"
        ]
        
        y_offset = 10
        for text in info_text:
            if text:
                surface = font.render(text, True, BLACK)
                SCREEN.blit(surface, (10, y_offset))
            y_offset += 30

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()
    
def draw_cartpole(states, dt):
    # 初始化 pygame
    pygame.init()

    # 窗口设置
    WIDTH, HEIGHT = 1200, 600
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cartpole Visualization")

    # 颜色定义
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)

    # 物理参数
    CART_WIDTH = 60
    CART_HEIGHT = 30
    POLE_LENGTH = 150  # 像素长度，对应 l=1.0m
    TRACK_HEIGHT = 200  # 轨道在屏幕上的垂直位置
    SCALE = 10  # 将物理单位（米）转换为像素

    # 时间参数
    FPS = 60
    frame_time = 1.0 / FPS  # 每帧的时间间隔（秒）

    # 主循环
    clock = pygame.time.Clock()
    current_time = 0.0  # 当前时间（秒）
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    current_time = 0.0  # 重置

        if not paused:
            current_time += frame_time

        # 计算总时间长度
        total_time = len(states) * dt
        
        # 循环时间
        current_time = current_time % total_time

        # 计算当前时间对应的状态索引（浮点数）
        state_index = current_time / dt
        
        # 获取插值索引
        idx0 = int(state_index) % len(states)
        idx1 = (idx0 + 1) % len(states)
        
        # 计算插值系数
        alpha = state_index - int(state_index)
        
        # 线性插值状态
        state0 = states[idx0]
        state1 = states[idx1]
        state = state0 * (1 - alpha) + state1 * alpha

        # 清屏
        SCREEN.fill(WHITE)

        # 获取当前状态
        cart_pos = state[0]  # 小车位置（米）
        pole_angle = state[1]  # 摆的角度（弧度）

        # 计算屏幕坐标
        # 小车中心位置（转换为 Python float）
        cart_x = float(WIDTH // 2 + cart_pos * SCALE)
        cart_y = float(TRACK_HEIGHT)

        # 摆的末端位置（pole_angle 是相对于竖直向下的角度）
        pole_end_x = float(cart_x + POLE_LENGTH * np.sin(pole_angle))
        pole_end_y = float(cart_y + POLE_LENGTH * np.cos(pole_angle))

        # 绘制轨道
        pygame.draw.line(SCREEN, DARK_GRAY, (0, TRACK_HEIGHT), (WIDTH, TRACK_HEIGHT), 3)

        # 绘制小车
        cart_rect = pygame.Rect(
            cart_x - CART_WIDTH // 2,
            cart_y - CART_HEIGHT // 2,
            CART_WIDTH,
            CART_HEIGHT
        )
        pygame.draw.rect(SCREEN, BLUE, cart_rect)
        pygame.draw.rect(SCREEN, BLACK, cart_rect, 2)

        # 绘制摆（从小车中心到摆的末端）
        pygame.draw.line(SCREEN, BLACK, (cart_x, cart_y), (pole_end_x, pole_end_y), 4)

        # 绘制摆的末端（小球）
        pygame.draw.circle(SCREEN, RED, (int(pole_end_x), int(pole_end_y)), 8)

        # 显示信息
        font = pygame.font.Font(None, 36)
        info_text = [
            f"Time: {current_time:.3f} s / {total_time:.3f} s",
            f"State Index: {state_index:.2f} ({idx0} -> {idx1}, α={alpha:.2f})",
            f"Cart Position: {cart_pos:.3f} m",
            f"Pole Angle: {np.degrees(pole_angle):.2f}°",
            f"Cart Velocity: {state[2]:.3f} m/s",
            f"Pole Angular Velocity: {state[3]:.3f} rad/s",
            "",
            "Press SPACE to pause/resume",
            "Press R to reset"
        ]
        
        y_offset = 10
        for text in info_text:
            if text:
                surface = font.render(text, True, BLACK)
                SCREEN.blit(surface, (10, y_offset))
            y_offset += 30

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()


def test_pendulum_cartpole():

    x_init = np.array([np.pi * 0.9, 0.0])
    x = x_init
    states = []  # 保存所有状态
    dt = 0.1  # 状态之间的时间间隔（秒）

    for i in range(100):
        x = rk4(Pendulum.params, Pendulum.dynamics, x, np.array([0.0]), dt)
        # 转换为 numpy 数组并保存
        states.append(np.array(x))

    # 转换为 numpy 数组以便索引
    states = np.array(states)

    draw_pendulum(states, dt)


def test_draw_cartpole():
    x_init = np.array([0.0, np.pi * 0.9, 0.0, 0.0])
    x = x_init
    states = []  # 保存所有状态
    dt = 0.01  # 状态之间的时间间隔（秒）

    for i in range(500):
        x = rk4(CartPole.params, CartPole.dynamics, x, np.array([0.0]), dt)
        # 转换为 numpy 数组并保存
        states.append(np.array(x))

    # 转换为 numpy 数组以便索引
    states = np.array(states)

    draw_cartpole(states, dt)


if __name__ == "__main__":
    test_draw_cartpole()