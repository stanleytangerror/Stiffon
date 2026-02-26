#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]

mod math;
mod rbd2d;
mod draw2d;

use crate::math::*;
use crate::rbd2d::*;

fn main() {
    let origin = mvec!(0.0, 0.0);
    let pose = Transform2d::new(origin, 0.0);
    let geometry = Geometry::rectangle(mvec!(0.5, 0.5));
    let mut body = Body2d::new(
        1.0,
        1.0,
        Vec2::ZERO,
        0.0,
        pose,
        geometry,
    );
    let mut solver = Solver::new();
    solver.add_body(body);

    loop {
        solver.step(1.0);
    }
}

// use macroquad::prelude::*;

// #[macroquad::main("Macroquad 矩形示例")]
// async fn main() {
//     loop {
//         // 清除屏幕
//         clear_background(BLACK);
        
//         // 设置矩形属性
//         let rect_x = screen_width() * 0.5 - 50.0;  // 居中
//         let rect_y = screen_height() * 0.5 - 50.0;
//         let rect_width = 100.0;
//         let rect_height = 80.0;
        
//         // 绘制矩形
//         draw_rectangle(rect_x, rect_y, rect_width, rect_height, RED);
        
//         // 可以绘制边框
//         draw_rectangle_lines(rect_x, rect_y, rect_width, rect_height, 3.0, WHITE);
        
//         // 可以绘制填充矩形
//         draw_rectangle(rect_x + 150.0, rect_y, rect_width, rect_height, BLUE);
        
//         // 可以绘制圆角矩形
//         draw_rectangle_rounded(
//             rect_x - 150.0,
//             rect_y,
//             rect_width,
//             rect_height,
//             15.0,
//             15.0,
//             GREEN
//         );
        
//         // 显示下一帧
//         next_frame().await;
//     }
// }