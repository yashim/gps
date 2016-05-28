""" This file defines an environment for the Box2D PointMass simulator. """
import numpy as np
import Box2D as b2

from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape, b2_pi)
from framework import Framework
from math import sqrt
from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES


def create_car(world, offset, wheel_radius, density=1.0,
               scale=(1.0, 1.0), chassis_vertices=None):
    """ create Box2D car"""
    x_offset, y_offset = offset
    scale_x, scale_y = scale
    if chassis_vertices is None:
        chassis_vertices = [
            (1.5, 0),
            (3, 2.5),
            (2.8, 5.5),
            (1, 10),
            (-1, 10),
            (-2.8, 5.5),
            (-3, 2.5),
            (-1.5, 0)
        ]

    wheel_vertices = [
        (0.1, 0.2),
        (-0.1, 0.2),
        (-0.1, -0.2),
        (0.1, -0.2),
    ]

    chassis_vertices = [(scale_x * x, scale_y * y)
                        for x, y in chassis_vertices]
    radius_scale = sqrt(scale_x ** 2 + scale_y ** 2)
    wheel_radius *= radius_scale

    chassis = world.CreateDynamicBody(
        position=(x_offset, y_offset),
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(vertices=chassis_vertices),
            density=density,
        )
    )

    wheel_shape = b2.b2PolygonShape()
    wheel_shape.SetAsBox(0.5, 1.25)

    left_front_wheel = world.CreateDynamicBody(
        fixtures=b2FixtureDef(
            shape=wheel_shape,
            density=density,
        )
    )

    right_front_wheel = world.CreateDynamicBody(
        fixtures=b2FixtureDef(
            shape=wheel_shape,
            density=density,
        )
    )

    left_rear_wheel = world.CreateDynamicBody(
        fixtures=b2FixtureDef(
            shape=wheel_shape,
            density=density,
        )
    )

    right_rear_wheel = world.CreateDynamicBody(
        fixtures=b2FixtureDef(
            shape=wheel_shape,
            density=density,
        )
    )

    left_front_joint = world.CreateRevoluteJoint(
        bodyA=chassis,
        bodyB=left_front_wheel,
        localAnchorA=(-3, 8.5),
        localAnchorB=(0, 0),
        enableLimit=True,
        lowerAngle = 0,
        upperAngle = 0,
    )

    right_front_joint = world.CreateRevoluteJoint(
        bodyA=chassis,
        bodyB=right_front_wheel,
        localAnchorA=(3, 8.5),
        localAnchorB=(0, 0),
        enableLimit=True,
        lowerAngle = 0,
        upperAngle = 0,
    )

    left_rear_joint = world.CreateRevoluteJoint(
        bodyA=chassis,
        bodyB=left_rear_wheel,
        localAnchorA=(-3, 0.75),
        localAnchorB=(0, 0),
        enableLimit=True,
        lowerAngle = 0,
        upperAngle = 0,
    )

    right_rear_joint = world.CreateRevoluteJoint(
        bodyA=chassis,
        bodyB=right_rear_wheel,
        localAnchorA=(3, 0.75),
        localAnchorB=(0, 0),
        enableLimit=True,
        lowerAngle = 0,
        upperAngle = 0,
    )

    return chassis

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
        self.initial_angle = b2.b2_pi
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

        car = create_car(self.world, offset=(0.0, 1.0), wheel_radius=0.1, scale=(1, 1))
        self.car = car

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
        self.car.linearVelocity = (0,0)#(action[0], action[1])

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
