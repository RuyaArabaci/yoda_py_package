#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
import cvxpy as cp

class MPCControllerNode(Node):
    def _init_(self):
        super()._init_('MPCControllerNode')

        self.deviation = 0.0          # Mevcut sapma (x ekseni)
        self.heading_error = 0.0
        self.dt = 0.1                 # Zaman adımı (saniye)
        self.N = 10                   # Tahmin ufku (adım sayısı)
        self.v_const = 0.05           # Sabit doğrusal hız (m/s)
        self.max_angular = 0.5        # Maksimum direksiyon açısı (rad/s)
        self.lambda_weight = 1.0      # Kontrol çabası ağırlığı
        self.current_heading_error = 0.0


        # ROS Publisher/Subscriber
        self.publisher = self.create_publisher(Twist, '/yoda/cmd_vel', 10)
        self.subscription = self.create_subscription(Vector3, '/lane_deviation', self.deviation_callback, 10)

        # Kontrol döngüsü her 100 ms'de bir çalışır
        self.timer = self.create_timer(self.dt, self.mpc_control_loop)

    def deviation_callback(self, msg):
        self.deviation = msg.x
        self.heading_error = msg.y

    def mpc_control_loop(self):
        # MPC değişkenleri
        e = cp.Variable(self.N + 1)  # yanal sapma
        theta = cp.Variable(self.N + 1)   # Yön hatası
        w = cp.Variable(self.N)      # açısal hız

        # Başlangıç koşulu
        constraints = [e[0] == self.deviation, theta[0] == self.current_heading_error]

        # Model: e[t+1] = e[t] + dt * w[t] , theta[t+1] = theta[t] + dt * w[t]

        for t in range(self.N):
            constraints += [e[t+1] == e[t] + self.dt * w[t]]
            constraints += [theta[t+1] == theta[t] + self.dt * w[t]]
            constraints += [cp.abs(w[t]) <= self.max_angular]  # direksiyon sınırı

        # Amaç fonksiyonu: sapmayı ve kontrol çabasını minimize et
        cost = cp.sum_squares(e) + 0.5 * cp.sum_squares(w) + 0.5 * cp.sum_squares(theta) + 10.0 * cp.sum_squares(w[1:] - w[:-1])
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        self.get_logger().info(f"MPC w: {w.value}")

        # Geçerli çözüm varsa kullan
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
            w_opt = float(w.value[0])
        else:
            self.get_logger().warn("CVXPY çözüm üretmedi, angular hız sıfırlandı.")
            w_opt = 0.0

        # Twist mesajı oluştur
        cmd = Twist()
        cmd.linear.x = self.v_const
        cmd.angular.z = float(w_opt)  # Kamera yönüne göre işaret terslenebilir
        self.publisher.publish(cmd)

        # Loglama
        self.get_logger().info(f"[MPC] deviation: {self.deviation:.3f}, angular.z: {-w_opt:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = MPCControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '_main_':
    main()