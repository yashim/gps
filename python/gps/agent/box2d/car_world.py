""" This file defines an environment for the Box2D PointMass simulator. """
import numpy as np
import Box2D as b2

from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape, b2_pi)
# from Box2D import *
from framework import Framework
from math import sqrt, cos, sin
from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES


class TDGroundArea(object):
    """
    An area on the ground that the car can run over
    """

    def __init__(self, friction_modifier):
        self.friction_modifier = friction_modifier


class TDTire(object):

    def __init__(self, car, max_forward_speed=100.0,
                 max_backward_speed=-20, max_drive_force=150,
                 turn_torque=15, max_lateral_impulse=3,
                 dimensions=(0.5, 1.25), density=1.0,
                 position=(0, 0)):

        world = car.body.world

        self.current_traction = 1
        self.turn_torque = turn_torque
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.max_drive_force = max_drive_force
        self.max_lateral_impulse = max_lateral_impulse
        self.ground_areas = []

        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(box=dimensions, density=density)
        self.body.userData = {'obj': self}

    @property
    def forward_velocity(self):
        body = self.body
        current_normal = body.GetWorldVector((0, 1))
        return current_normal.dot(body.linearVelocity) * current_normal

    @property
    def lateral_velocity(self):
        body = self.body

        right_normal = body.GetWorldVector((1, 0))
        return right_normal.dot(body.linearVelocity) * right_normal

    def update_friction(self):
        impulse = -self.lateral_velocity * self.body.mass
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length

        self.body.ApplyLinearImpulse(self.current_traction * impulse,
                                     self.body.worldCenter, True)

        aimp = 0.1 * self.current_traction * \
            self.body.inertia * -self.body.angularVelocity
        self.body.ApplyAngularImpulse(aimp, True)

        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()

        drag_force_magnitude = -2 * current_forward_speed
        self.body.ApplyForce(self.current_traction * drag_force_magnitude * current_forward_normal,
                             self.body.worldCenter, True)

    def update_drive(self, keys):
        if 'up' in keys:
            desired_speed = self.max_forward_speed
        elif 'down' in keys:
            desired_speed = self.max_backward_speed
        else:
            return

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)

        # apply necessary force
        force = 0.0
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(self.current_traction * force * current_forward_normal,
                             self.body.worldCenter, True)

    def update_turn(self, keys):
        if 'left' in keys:
            desired_torque = self.turn_torque
        elif 'right' in keys:
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)

    def add_ground_area(self, ud):
        if ud not in self.ground_areas:
            self.ground_areas.append(ud)
            self.update_traction()

    def remove_ground_area(self, ud):
        if ud in self.ground_areas:
            self.ground_areas.remove(ud)
            self.update_traction()

    def update_traction(self):
        if not self.ground_areas:
            self.current_traction = 1
        else:
            self.current_traction = 0
            mods = [ga.friction_modifier for ga in self.ground_areas]

            max_mod = max(mods)
            if max_mod > self.current_traction:
                self.current_traction = max_mod


class TDCar(object):
    vertices = [(1.5, 0.0),
                (3.0, 2.5),
                (2.8, 5.5),
                (1.0, 10.0),
                (-1.0, 10.0),
                (-2.8, 5.5),
                (-3.0, 2.5),
                (-1.5, 0.0),
                ]

    tire_anchors = [(-3.0, 0.75),
                    (3.0, 0.75),
                    (-3.0, 8.50),
                    (3.0, 8.50),
                    ]

    def __init__(self, world, vertices=None,
                 tire_anchors=None, density=0.1, position=(0, 0),
                 **tire_kws):
        if vertices is None:
            vertices = TDCar.vertices

        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(vertices=vertices, density=density)
        self.body.userData = {'obj': self}

        self.tires = [TDTire(self, **tire_kws) for i in range(4)]

        if tire_anchors is None:
            anchors = TDCar.tire_anchors

        joints = self.joints = []
        for tire, anchor in zip(self.tires, anchors):
            j = world.CreateRevoluteJoint(bodyA=self.body,
                                          bodyB=tire.body,
                                          localAnchorA=anchor,
                                          # center of tire
                                          localAnchorB=(0, 0),
                                          enableMotor=False,
                                          maxMotorTorque=1000,
                                          enableLimit=True,
                                          lowerAngle=0,
                                          upperAngle=0,
                                          )

            tire.body.position = self.body.worldCenter + anchor
            joints.append(j)

    def update(self, keys, hz):
        for tire in self.tires:
            tire.update_friction()

        for tire in self.tires:
            tire.update_drive(keys)

        # control steering
        lock_angle = math.radians(40.)
        # from lock to lock in 0.5 sec
        turn_speed_per_sec = math.radians(160.)
        turn_per_timestep = turn_speed_per_sec / hz
        desired_angle = 0.0

        if 'left' in keys:
            desired_angle = lock_angle
        elif 'right' in keys:
            desired_angle = -lock_angle

        front_left_joint, front_right_joint = self.joints[2:4]
        angle_now = front_left_joint.angle
        angle_to_turn = desired_angle - angle_now

        # TODO fix b2Clamp for non-b2Vec2 types
        if angle_to_turn < -turn_per_timestep:
            angle_to_turn = -turn_per_timestep
        elif angle_to_turn > turn_per_timestep:
            angle_to_turn = turn_per_timestep

        new_angle = angle_now + angle_to_turn
        # Rotate the tires by locking the limits:
        front_left_joint.SetLimits(new_angle, new_angle)
        front_right_joint.SetLimits(new_angle, new_angle)


# def create_car(world, offset, wheel_radius, density=1.0,
#                scale=(1.0, 1.0), chassis_vertices=None):
#     """ create Box2D car"""
#     x_offset, y_offset = offset
#     scale_x, scale_y = scale
#     if chassis_vertices is None:
#         chassis_vertices = [
#             (1.5, 0),
#             (3, 2.5),
#             (2.8, 5.5),
#             (1, 10),
#             (-1, 10),
#             (-2.8, 5.5),
#             (-3, 2.5),
#             (-1.5, 0)
#         ]

#     wheel_vertices = [
#         (0.1, 0.2),
#         (-0.1, 0.2),
#         (-0.1, -0.2),
#         (0.1, -0.2),
#     ]

#     chassis_vertices = [(scale_x * x, scale_y * y)
#                         for x, y in chassis_vertices]
#     radius_scale = sqrt(scale_x ** 2 + scale_y ** 2)
#     wheel_radius *= radius_scale

#     chassis = world.CreateDynamicBody(
#         position=(x_offset, y_offset),
#         fixtures=b2FixtureDef(
#             shape=b2PolygonShape(vertices=chassis_vertices),
#             density=density,
#         )
#     )

#     wheel_shape = b2.b2PolygonShape()
#     wheel_shape.SetAsBox(0.5, 1.25)

#     left_front_wheel = world.CreateDynamicBody(
#         fixtures=b2FixtureDef(
#             shape=wheel_shape,
#             density=density,
#         )
#     )

#     right_front_wheel = world.CreateDynamicBody(
#         fixtures=b2FixtureDef(
#             shape=wheel_shape,
#             density=density,
#         )
#     )

#     left_rear_wheel = world.CreateDynamicBody(
#         fixtures=b2FixtureDef(
#             shape=wheel_shape,
#             density=density,
#         )
#     )

#     right_rear_wheel = world.CreateDynamicBody(
#         fixtures=b2FixtureDef(
#             shape=wheel_shape,
#             density=density,
#         )
#     )

#     left_front_joint = world.CreateRevoluteJoint(
#         bodyA=chassis,
#         bodyB=left_front_wheel,
#         localAnchorA=(-3, 8.5),
#         localAnchorB=(0, 0),
#         enableLimit=True,
#         lowerAngle = 0,
#         upperAngle = 0,
#     )

#     right_front_joint = world.CreateRevoluteJoint(
#         bodyA=chassis,
#         bodyB=right_front_wheel,
#         localAnchorA=(3, 8.5),
#         localAnchorB=(0, 0),
#         enableLimit=True,
#         lowerAngle = 0,
#         upperAngle = 0,
#     )

#     left_rear_joint = world.CreateRevoluteJoint(
#         bodyA=chassis,
#         bodyB=left_rear_wheel,
#         localAnchorA=(-3, 0.75),
#         localAnchorB=(0, 0),
#         enableLimit=True,
#         lowerAngle = 0,
#         upperAngle = 0,
#     )

#     right_rear_joint = world.CreateRevoluteJoint(
#         bodyA=chassis,
#         bodyB=right_rear_wheel,
#         localAnchorA=(3, 0.75),
#         localAnchorB=(0, 0),
#         enableLimit=True,
#         lowerAngle = 0,
#         upperAngle = 0,
#     )

#     return chassis

class CarWorld(Framework):
    """ This class defines the point mass and its environment."""
    name = "Car"

    def __init__(self, x0, target, render):
        self.render = render
        if self.render:
            super(CarWorld, self).__init__()
        else:
            self.world = b2.b2World(gravity=(0, 0), doSleep=True)
        self.world.gravity = (0.0, 0.0)
        self.initial_position = (x0[0], x0[1])
        self.initial_angle = b2.b2_pi / 4
        self.initial_linear_velocity = (x0[2], x0[3])
        self.initial_angular_velocity = 0

        ground = self.world.CreateBody(position=(0, 20))
        ground.CreateEdgeChain(
            [(-20, -20),
             (-20, 20),
             (20, 20),
             (20, -20),
             (-20, -20)]
            )

        xf1 = b2.b2Transform()
        xf1.angle = 0.3524 * b2.b2_pi
        xf1.position = b2.b2Mul(xf1.R, (1.0, 0.0))

        xf2 = b2.b2Transform()
        xf2.angle = -0.3524 * b2.b2_pi
        xf2.position = b2.b2Mul(xf2.R, (-1.0, 0.0))

        # car = create_car(self.world, offset=(0.0, 1.0), wheel_radius=0.1, scale=(1, 1))
        # self.car = car

        # self.car = self.world.CreateDynamicBody(
        #     position=self.initial_position,
        #     angle=self.initial_angle,
        #     linearVelocity=self.initial_linear_velocity,
        #     angularVelocity=self.initial_angular_velocity,
        #     angularDamping=5,
        #     linearDamping=0.1,
        #     shapes=[b2.b2PolygonShape(vertices=[xf1*(-1, 0),
        #                                         xf1*(1, 0), xf1*(0, .5)]),
        #             b2.b2PolygonShape(vertices=[xf2*(-1, 0),
        #                                         xf2*(1, 0), xf2*(0, .5)])],
        #     shapeFixture=b2.b2FixtureDef(density=1.0),
        # )

        self.car = TDCar(self.world)



        self.target = self.world.CreateStaticBody(
            position=target[:2],
            angle=self.initial_angle,
            shapes=[b2.b2PolygonShape(vertices=[xf1*(-1, 0), xf1*(1, 0),
                                                xf1*(0, .5)]),
                    b2.b2PolygonShape(vertices=[xf2*(-1, 0), xf2*(1, 0),
                                                xf2*(0, .5)])],
        )
        self.target.active = False

    def run(self):
        """Initiates the first time step
        """
        if self.render:
            super(CarWorld, self).run()
        else:
            self.run_next(None)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(CarWorld, self).run_next(action)
        else:
            if action is not None:
                self.car.linearVelocity = (0,0)# (action[0], action[1])
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Called upon every step. """
        # a = self.car.angle
        # vel_x = cos(a) * 0 + sin(a) * action[0]
        # vel_y = -sin(a) * 0 + cos(a) * action[0]
        # self.car.linearVelocity = (vel_x, vel_y)#(0,0)
        # print self.car.linearVelocity, self.car.angle
        # print self.car.GetLocalPoint((0, 1))
        # vec = self.car.GetWorldPoint((action[0], 0))
        # vec.Normalize()
        
        self.car.linearVelocity = (0,0)
        # self.car.angularVelocity = action[1]
        super(CarWorld, self).Step(settings)

    def reset_world(self):
        """ This resets the world to its initial state"""
        self.world.ClearForces()
        self.car.position = self.initial_position
        self.car.angle = self.initial_angle
        self.car.angularVelocity = self.initial_angular_velocity
        self.car.linearVelocity = self.initial_linear_velocity

    def get_state(self):
        """ This retrieves the state of the point mass"""
        state = {END_EFFECTOR_POINTS: np.append(np.array(self.car.position), [0]),
                 END_EFFECTOR_POINT_VELOCITIES: np.append(np.array(self.car.linearVelocity), [0])}

        return state
