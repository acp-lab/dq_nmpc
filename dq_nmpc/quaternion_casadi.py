import dataclasses
from typing import List
import casadi as cs
from dq_nmpc.type import Scalar, Vector
import numpy as np
from numbers import Number

@dataclasses.dataclass
class Quaternion():
    # Properties of the class
    q: Vector # Property to store the quaternion vector
    def __init__(self, q = None):
        """
        Constructor method for the Quaternion class.

        Parameters:
        - q: Vector representing the quaternion. Can be a NumPy array or a CasADi MX or SX object.

        Raises:
        - ValueError: If the quaternion vector does not have exactly 4 elements.
        - TypeError: If the input is not a NumPy array or a CasADi MX or SX object.
        """
        if q is not None:
            # Check if the variable is a NumPy array
            if isinstance(q, np.ndarray): 
                if q.shape[0] != 4:
                    raise ValueError("quaternion must have exactly 4 elements.")
                q_aux = q.reshape((4, 1))
                self.q = q_aux

            # Check for CasADi variables
            elif isinstance(q, cs.MX) or isinstance(q, cs.SX):
                if q.shape[0] != 4:
                    raise ValueError("quaternion must have exactly 4 elements.")
                self.q = q
            else:
                raise TypeError("quaternion must be an ndarray or Casadi MX or SX")

    def __getattr__(self, attr):
        """
        Custom __getattr__ method for the Quaternion class.
        This method enables access to the same attributes inside numpy objects or CasADi MX or SX objects.
        """
        return getattr(self.q, attr)

    def __repr__(self) -> str:
        """
        Custom __repr__ method for the Quaternion class.
        This method returns a string representation of the Quaternion object.
        """
        return f"Quaternion: {self.q}"

    def __str__(self) -> str:
        """
        Custom __str__ method for the Quaternion class.
        This method returns a string representation of the Quaternion object.
        """
        return f'Quaternion(: {self.q})'


    @property
    def get(self) -> Vector:
        """
        Property method for the Quaternion class.
        This property allows accessing the underlying quaternion vector.
        """
        return self.q

    def __mul__(self, q2: "Quaternion") -> "Quaternion":
        """
        Custom __mul__ method for the Quaternion class.
        This method performs multiplication between quaternions.

        Parameters:
        - q2: The quaternion to be multiplied with the current quaternion.

        Returns:
        - A new Quaternion object representing the result of the multiplication.

        Raises:
        - TypeError: If the input is not a Quaternion or a scalar.

        Note:
        This method supports multiplication between quaternions and scalars.
        If q2 is a Quaternion, the result is the product of the quaternions.
        If q2 is a scalar, the result is the quaternion scaled by the scalar.

        Example:
        If q1 and q2 are Quaternion objects:
        - q1 * q2 will return the product of q1 and q2 (Quaternion object).
        - q1 * scalar will return q1 scaled by the scalar (Quaternion object).
        """
        if isinstance(q2, Quaternion):
            return Quaternion(q = Quaternion.product(self.q, q2.q))

        elif isinstance(q2, Number):
            q = self.q
            q_out = q * q2
            return Quaternion(q = q_out)
        elif (isinstance(q2, cs.MX) and isinstance(self.q, cs.MX)):
            if q2.shape[0]==1:
                q = self.q
                q_out = q * q2
                return Quaternion(q = q_out)
            else:
                raise TypeError("Left Multiplication is only defined for Quaternions and scalars of the same type")

        elif (isinstance(q2, cs.SX) and isinstance(self.q, cs.SX)):
            if q2.shape[0]==1:
                q = self.q
                q_out = q * q2
                return Quaternion(q = q_out)
            else:
                raise TypeError("Left Multiplication is only defined for Quaternions and scalars of the smae type")
        else:
            raise TypeError("Left Multiplication is only defined for Quaternions and scalars")

    def __rmul__(self, q2) -> "Quaternion":
        """
        Custom __rmul__ method for the Quaternion class.
        This method performs left multiplication of a scalar with the current quaternion.

        Parameters:
        - q2: The scalar to be left-multiplied with the current quaternion.

        Returns:
        - A new Quaternion object representing the result of the left multiplication.

        Raises:
        - TypeError: If the input is not a scalar.

        Note:
        This method supports left multiplication of the quaternion by a scalar.

        Example:
        If q is a Quaternion object and scalar is a scalar value:
        - scalar * q will return the quaternion q scaled by the scalar (Quaternion object).
        """
        if isinstance(q2, Number):
            q = self.q
            q_out =  q2 * q
            return Quaternion(q = q_out)
        elif (isinstance(q2, cs.MX) and isinstance(self.q, cs.MX)):
            if q2.shape[0]==1:
                q = self.q
                q_out =  q2 * q
                return Quaternion(q = q_out)
            else:
                raise TypeError("Right Multiplication is only defined for scalars of the same type")

        elif (isinstance(q2, cs.SX) and isinstance(self.q, cs.SX)):
            if q2.shape[0]==1:
                q = self.q
                q_out =  q2 * q
                return Quaternion(q = q_out)
            else:
                raise TypeError("Right Multiplication is only defined for scalars of the same type")
        else:
            raise TypeError("Right Multiplication is only defined for scalars of the same type")

    @staticmethod
    def product(p: Vector, q: Vector) -> Vector:
        """
        Static method to compute the product of two quaternions.

        Parameters:
        - p: Vector representing the first quaternion.
        - q: Vector representing the second quaternion.

        Returns:
        - Vector representing the product of the two quaternions.

        Raises:
        - TypeError: If the elements of both quaternions are not of the same type.

        Note:
        This method supports the computation of quaternion products for NumPy arrays, CasADi MX, and CasADi SX objects.
        It performs different operations based on the type of the input quaternions.

        Example:
        If p and q are vectors representing quaternions:
        - Quaternion.product(p, q) will return the product of the quaternions.
        """
        if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            #aux_1 = p[0, 0] * q[0, 0] - np.dot(p[1:4, 0], q[1:4, 0])
            #aux_2 = p[0, 0] * q[1:4, 0] + q[0, 0]* p[1:4, 0]+ np.cross(p[1:4, 0], q[1:4, 0])
            #q_product = np.vstack((aux_1, aux_2[0], aux_2[1], aux_2[2]))
            #q_product = q_product.reshape((4, 1))

            # New Product
            H_plus = np.array([[p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]],
                               [p[1, 0], p[0, 0], -p[3, 0], p[2, 0]],
                               [p[2, 0], p[3, 0], p[0, 0], -p[1, 0]],
                               [p[3, 0], -p[2, 0], p[1, 0], p[0, 0]]])
            q_product = H_plus@q
            q_product = q_product.reshape((4, 1))
            return q_product

        elif isinstance(p, cs.MX) and isinstance(q, cs.MX):
            #aux_1 = p[0, 0] * q[0, 0] - cs.dot(p[1:4, 0], q[1:4, 0])
            #aux_2 = p[0, 0] * q[1:4, 0] + q[0, 0]* p[1:4, 0]+ cs.cross(p[1:4, 0], q[1:4, 0])
            #q_product = cs.vertcat(aux_1, aux_2)
            H_plus = cs.vertcat(cs.horzcat(p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]),
                                cs.horzcat(p[1, 0], p[0, 0], -p[3, 0], p[2, 0]),
                                cs.horzcat(p[2, 0], p[3, 0], p[0, 0], -p[1, 0]),
                                cs.horzcat(p[3, 0], -p[2, 0], p[1, 0], p[0, 0]))
            q_product = H_plus@q
            return q_product

        elif isinstance(p, cs.SX) and isinstance(q, cs.SX):
            #aux_1 = p[0, 0] * q[0, 0] - cs.dot(p[1:4, 0], q[1:4, 0])
            #aux_2 = p[0, 0] * q[1:4, 0] + q[0, 0]* p[1:4, 0]+ cs.cross(p[1:4, 0], q[1:4, 0])
            #q_product = cs.vertcat(aux_1, aux_2)
            H_plus = cs.vertcat(cs.horzcat(p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]),
                                cs.horzcat(p[1, 0], p[0, 0], -p[3, 0], p[2, 0]),
                                cs.horzcat(p[2, 0], p[3, 0], p[0, 0], -p[1, 0]),
                                cs.horzcat(p[3, 0], -p[2, 0], p[1, 0], p[0, 0]))
            q_product = H_plus@q
            return q_product
        else:
            raise TypeError("The elements of both quaternions should be of the same type.")

    def __add__(self, q2: "Quaternion") -> "Quaternion":
        """
        Custom __add__ method for the Quaternion class.
        This method performs addition between quaternions.

        Parameters:
        - q2: The quaternion to be added with the current quaternion.

        Returns:
        - A new Quaternion object representing the result of the addition.

        Raises:
        - TypeError: If the input is not a Quaternion or a scalar.

        Note:
        This method supports addition between quaternions and scalars.
        If q2 is a Quaternion, the result is the sum of the quaternions.
        If q2 is a scalar or a CasADi MX or SX object, the result is the quaternion incremented by the value.

        Example:
        If q1 and q2 are Quaternion objects:
        - q1 + q2 will return the sum of q1 and q2 (Quaternion object).
        - q1 + scalar will return q1 incremented by the scalar (Quaternion object).
        """
        if isinstance(q2, Quaternion):
            return Quaternion(q = Quaternion.add(self.q, q2.q))

        elif (isinstance(q2, Number)):
            q = self.q
            q_out = q + q2
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.MX) and isinstance(self.q, cs.MX)):
            q = self.q
            q_out = q + q2
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.SX) and isinstance(self.q, cs.SX)):
            q = self.q
            q_out = q + q2
            return Quaternion(q = q_out)
        else:
            raise TypeError("Right addition is only defined for Quaternions and scalars of the same type.")

    def __radd__(self, q2: "Quaternion") -> "Quaternion":
        """
        Custom __radd__ method for the Quaternion class.
        This method performs left addition of a scalar or quaternion with the current quaternion.

        Parameters:
        - q2: The scalar or quaternion to be left-added with the current quaternion.

        Returns:
        - A new Quaternion object representing the result of the left addition.

        Raises:
        - TypeError: If the input is not a Quaternion or a scalar.

        Note:
        This method supports left addition of a quaternion or a scalar with the current quaternion.

        Example:
        If q is a Quaternion object and scalar is a scalar value:
        - scalar + q will return the quaternion q incremented by the scalar (Quaternion object).
        - q1 + q2 will return the sum of q1 and q2 (Quaternion object).
        """
        if isinstance(q2, Quaternion):
            return Quaternion(q = Quaternion.add(q2.q, self.q))

        elif (isinstance(q2, Number)):
            q = self.q
            q_out = q2 + q
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.MX) and isinstance(self.q, cs.MX)):
            q = self.q
            q_out = q2 + q
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.SX) and isinstance(self.q, cs.SX)):
            q = self.q
            q_out = q2 + q
            return Quaternion(q = q_out)

        else:
            raise TypeError("Left add only is defined for Quaternions and scalars")

    @staticmethod
    def add(p: Vector, q: Vector) -> Vector:
        """
        Static method to compute the addition of two quaternions.

        Parameters:
        - p: Vector representing the first quaternion.
        - q: Vector representing the second quaternion.

        Returns:
        - Vector representing the sum of the two quaternions.

        Raises:
        - TypeError: If the elements of both quaternions are not of the same type.

        Note:
        This method supports addition of quaternions represented as NumPy arrays, CasADi MX, and CasADi SX objects.

        Example:
        If p and q are vectors representing quaternions:
        - Quaternion.add(p, q) will return the sum of the quaternions.
        """
        if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            aux_1 = p + q
            q_product = aux_1
            return q_product

        elif isinstance(p, cs.MX) and isinstance(q, cs.MX):
            aux_1 = p + q
            q_product = aux_1
            return q_product

        elif isinstance(p, cs.SX) and isinstance(q, cs.SX):
            aux_1 = p + q
            q_product = aux_1
            return q_product
        else:
            raise TypeError("The elements of both quaternions should be of the same type.")
            
    def __sub__(self, q2: "Quaternion") -> "Quaternion":
        """
        Custom __sub__ method for the Quaternion class.
        This method performs subtraction between quaternions.

        Parameters:
        - q2: The quaternion to be subtracted from the current quaternion.

        Returns:
        - A new Quaternion object representing the result of the subtraction.

        Raises:
        - TypeError: If the input is not a Quaternion or a scalar.

        Note:
        This method supports subtraction between quaternions and scalars.
        If q2 is a Quaternion, the result is the difference between the quaternions.
        If q2 is a scalar or a CasADi MX or SX object, the result is the quaternion decremented by the value.

        Example:
        If q1 and q2 are Quaternion objects:
        - q1 - q2 will return the difference between q1 and q2 (Quaternion object).
        - q1 - scalar will return q1 decremented by the scalar (Quaternion object).
        """
        if isinstance(q2, Quaternion):
            return Quaternion(q = Quaternion.sub(self.q, q2.q))

        elif (isinstance(q2, Number)):
            q = self.q
            q_out =  q - q2
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.MX) and isinstance(self.q, cs.MX)):
            q = self.q
            q_out =  q - q2
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.SX) and isinstance(self.q, cs.SX)):
            q = self.q
            q_out =  q - q2
            return Quaternion(q = q_out)

        else:
            raise TypeError("Right sub only defined for Quaternions and scalars")

    def __rsub__(self, q2: "Quaternion") -> "Quaternion":
        if isinstance(q2, Quaternion):
            return Quaternion(q = Quaternion.sub(q2.q, self.q))

        elif (isinstance(q2, Number)):
            q = self.q
            q_out =  q2 - q
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.MX) and isinstance(self.q, cs.MX)):
            q = self.q
            q_out =  q2 - q
            return Quaternion(q = q_out)

        elif (isinstance(q2, cs.SX) and isinstance(self.q, cs.SX)):
            q = self.q
            q_out =  q2 - q
            return Quaternion(q = q_out)

        else:
            raise TypeError("Left sub only defined for Quaternions and scalars")

    @staticmethod
    def sub(p: Vector, q: Vector) -> Vector:
        # Function uses to define the substraction operation
        if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            aux_1 = p - q
            q_product = aux_1

        elif isinstance(p, cs.MX) and isinstance(q, cs.MX):
            aux_1 = p - q
            q_product = aux_1

        elif isinstance(p, cs.SX) and isinstance(q, cs.SX):
            aux_1 = p - q
            q_product = aux_1
        else:
            raise TypeError("The elements of both quaternions should be of the same type.")
        return q_product

    def conjugate(self) -> "Quaternion":
        q = self.q
        if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            qw = q[0, 0]
            qx = -q[1, 0]
            qy = -q[2, 0]
            qz = -q[3, 0]
            q_conjugate = np.vstack((qw, qx, qy, qz))
            # Funtion that defines the conjugate of a quaternion
            return Quaternion(q = q_conjugate)
        elif isinstance(q, cs.MX):
            qw = q[0, 0]
            qx = -q[1, 0]
            qy = -q[2, 0]
            qz = -q[3, 0]
            q_conjugate = cs.vertcat(qw, qx, qy, qz)
            # Funtion that defines the conjugate of a quaternion
            return Quaternion(q = q_conjugate)
        elif isinstance(q, cs.SX):
            qw = q[0, 0]
            qx = -q[1, 0]
            qy = -q[2, 0]
            qz = -q[3, 0]
            q_conjugate = cs.vertcat(qw, qx, qy, qz)
            # Funtion that defines the conjugate of a quaternion
            return Quaternion(q = q_conjugate)
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")
    @property
    def norm(self) -> "Scalar":
        q = self.q
        if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            norm = np.sqrt(np.dot(q.T, q))
            norm_value = norm[0,0]
            return norm_value
        elif isinstance(q, cs.MX):
            norm = cs.sqrt(cs.dot(q, q))
            norm_value = norm
            return norm_value
        elif isinstance(q, cs.SX):
            norm = cs.sqrt(cs.dot(q, q))
            norm_value = norm
            return norm_value
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")

    @property
    def square_norm(self) -> "Scalar":
        q = self.q
        if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            norm = np.dot(q.T, q)
            norm_value = norm[0,0]
            return norm_value
        elif isinstance(q, cs.MX):
            norm = cs.dot(q, q)
            norm_value = norm
            return norm_value
        elif isinstance(q, cs.SX):
            norm = cs.dot(q, q)
            norm_value = norm
            return norm_value
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")

    def inverse(self) -> "Quaternion":
        # Function that computes the inverse of a quaternion
        return self.conjugate() / self.square_norm()

    def set(self, q = None):
         if q is not None:
            # Check if the vairbale is a np.array
            if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
                if q.shape[0] != 4:
                    raise ValueError("quaternion must have exactly 4 elements.")
                q_aux = q.reshape((4, 1))
                self.q = q_aux

            # Check for casadi variables
            elif isinstance(q, cs.MX) or isinstance(q, cs.SX):
                if q.shape[0] != 4:
                    raise ValueError("quaternion must have exactly 4 elements.")
                self.q = q
            else:
                raise TypeError("quaternion must be an ndarray or Casadi MX  SX")
    #@property
    def angle_axis(self):
        q = self.q
        norm = self.norm
        if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            qw = q[0, 0]
            angle = np.arctan2(norm, qw)
            if  norm > 0.0:
                x = q[1, 0] / norm
                y = q[2, 0] / norm
                z = q[3, 0] / norm
            else:
                angle = 0.0
                x = 0.0
                y = 0.0
                z = 1.0
            result = np.vstack((angle, x, y, z))
            return result
            #if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            #if  np.isclose(qw, 1, atol=1.e-12):
            #    angle = 0.0
            #    x = 0.0
            #    y = 0.0
            #    z = 1.0
            #else:
            #    angle = 2.0*np.arccos(qw)
            #    x = q[1, 0] / np.sqrt(1 - qw*qw)
            #    y = q[2, 0] / np.sqrt(1 - qw*qw)
            #    z = q[3, 0] / np.sqrt(1 - qw*qw)
            #result = np.vstack((angle, x, y, z))
            #return result
        elif isinstance(q, cs.MX):
            #qw = q[0, 0]
            #angle = cs.atan2(norm, qw)
            # Define conditions
            #condition1 = norm > 0.0

            # Define expressions for each condition
            #expr1 =  cs.vertcat(angle, q[1, 0]/norm, q[2, 0]/norm, q[3, 0]/norm)
            #expr2 = cs.vertcat(angle, 0.0, 0.0, 1.0)

            # Nested if_else to implement multiple branches
            #result = cs.if_else(condition1, expr1, expr2) 
            norm = cs.norm_2(q[1:4] + cs.np.finfo(np.float64).eps)
            angle = cs.atan2(norm, q[0])
            expr1 =  cs.vertcat(angle, q[1, 0]/norm, q[2, 0]/norm, q[3, 0]/norm)
            result = expr1
            return result
        elif isinstance(q, cs.SX):
            #qw = q[0, 0]
            #angle = cs.atan2(norm, qw)
            # Define conditions
            #condition1 = norm > 0.0

            # Define expressions for each condition
            #expr1 =  cs.vertcat(angle, q[1, 0]/norm, q[2, 0]/norm, q[3, 0]/norm)
            #expr2 = cs.vertcat(angle, 0.0, 0.0, 1.0)

            # Nested if_else to implement multiple branches
            #result = cs.if_else(condition1, expr1, expr2) 
            norm = cs.norm_2(q[1:4] + cs.np.finfo(np.float64).eps)
            angle = cs.atan2(norm, q[0])
            expr1 =  cs.vertcat(angle, q[1, 0]/norm, q[2, 0]/norm, q[3, 0]/norm)
            result = expr1
            return result
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")

    def ln(self):
        # Log mapping
        q = self.q
        angle_axis_aux = self.angle_axis()
        if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            angle = angle_axis_aux[0, 0]
            x = angle_axis_aux[1, 0]
            y = angle_axis_aux[2, 0]
            z = angle_axis_aux[3, 0]
            result = np.vstack((0.0, (1/2)*angle*x, (1/2)*angle*y, (1/2)*angle*z))
            return Quaternion(q = result)
        elif isinstance(q, cs.MX):
            angle = angle_axis_aux[0, 0]
            x = angle_axis_aux[1, 0]
            y = angle_axis_aux[2, 0]
            z = angle_axis_aux[3, 0]
            result = cs.vertcat(0.0, (1/2)*angle*x, (1/2)*angle*y, (1/2)*angle*z)
            return Quaternion(q = result)
        elif isinstance(q, cs.SX):
            angle = angle_axis_aux[0, 0]
            x = angle_axis_aux[1, 0]
            y = angle_axis_aux[2, 0]
            z = angle_axis_aux[3, 0]
            result = cs.vertcat(0.0, (1/2)*angle*x, (1/2)*angle*y, (1/2)*angle*z)
            return Quaternion(q = result)
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")

    def ln_trans(self):
        # Log mapping
        q = self.q
        angle_axis_aux = self.angle_axis
        if isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            angle = q[0, 0]
            x = q[1, 0]
            y = q[2, 0]
            z = q[3, 0]
            result = np.vstack((0.0, (1/2)*x, (1/2)*y, (1/2)*z))
            return Quaternion(q = result)
        elif isinstance(q, cs.MX):
            angle = q[0, 0]
            x = q[1, 0]
            y = q[2, 0]
            z = q[3, 0]
            result = cs.vertcat(0.0, (1/2)*x, (1/2)*y, (1/2)*z)
            return Quaternion(q = result)
        elif isinstance(q, cs.SX):
            angle = q[0, 0]
            x = q[1, 0]
            y = q[2, 0]
            z = q[3, 0]
            result = cs.vertcat(0.0, (1/2)*x, (1/2)*y, (1/2)*z)
            return Quaternion(q = result)
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")

    def vector_dot_product(self, q2):
        if isinstance(q2, Quaternion):
            q_rot = Quaternion.vector_dot(self.q, q2.q)
            return Quaternion(q = q_rot)
        else:
            raise TypeError("Vector Dot Product only defined for Quaternions")

    @staticmethod
    def vector_dot(p: Vector, q: Vector) -> Vector:
        # Funtion that defines the addition opperation
        if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            aux_1 = p * q
            q_product = aux_1
            return q_product
        elif isinstance(p, cs.MX) and isinstance(q, cs.MX):
            aux_1 = p * q
            q_product = aux_1
            return q_product
        elif isinstance(p, cs.SX) and isinstance(q, cs.SX):
            aux_1 = p * q
            q_product = aux_1
            return q_product
        else:
            raise TypeError("The elements of both quaternions should be of the same type.")

    def cross(self, p: "Quaternion") -> "Quaternion":
        q = self.q
        p = p.get
        if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):  # Use Vector directly without parentheses
            product = np.cross(q[1:4, 0], p[1:4, 0])
            result = np.vstack((0.0, product[0], product[1], product[2]))
            return Quaternion(q = result)

        elif isinstance(p, cs.MX) and isinstance(q, cs.MX):
            aux = cs.MX([0])
            product = cs.vertcat(aux[0], q[2]*p[3] - q[3]*p[2], q[3]*p[1] - q[1]*p[3], q[1]*p[2] - q[2]*p[1])
            return Quaternion(q = product)

        elif isinstance(p, cs.SX) and isinstance(q, cs.SX):
            aux = cs.SX([0])
            product = cs.vertcat(aux[0], q[2]*p[3] - q[3]*p[2], q[3]*p[1] - q[1]*p[3], q[1]*p[2] - q[2]*p[1])
            return Quaternion(q = product)
        else:
            raise TypeError("The elements of both quaternions should be of the same type.")

    @property
    def H_plus(self) -> "Vector":
        p = self.q
        if isinstance(p, np.ndarray):  # Use Vector directly without parentheses
            H_plus = np.array([[p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]],
                               [p[1, 0], p[0, 0], -p[3, 0], p[2, 0]],
                               [p[2, 0], p[3, 0], p[0, 0], -p[1, 0]],
                               [p[3, 0], -p[2, 0], p[1, 0], p[0, 0]]])
            return H_plus
        elif isinstance(p, cs.MX):
            H_plus = cs.vertcat(cs.horzcat(p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]),
                                cs.horzcat(p[1, 0], p[0, 0], -p[3, 0], p[2, 0]),
                                cs.horzcat(p[2, 0], p[3, 0], p[0, 0], -p[1, 0]),
                                cs.horzcat(p[3, 0], -p[2, 0], p[1, 0], p[0, 0]))
            return H_plus
        elif isinstance(p, cs.SX):
            H_plus = cs.vertcat(cs.horzcat(p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]),
                                cs.horzcat(p[1, 0], p[0, 0], -p[3, 0], p[2, 0]),
                                cs.horzcat(p[2, 0], p[3, 0], p[0, 0], -p[1, 0]),
                                cs.horzcat(p[3, 0], -p[2, 0], p[1, 0], p[0, 0]))
            return H_plus
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")

    @property
    def H_minus(self) -> "Vector":
        p = self.q
        if isinstance(p, np.ndarray):  # Use Vector directly without parentheses
            H_minus = np.array([[p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]],
                               [p[1, 0], p[0, 0], p[3, 0], -p[2, 0]],
                               [p[2, 0], -p[3, 0], p[0, 0], p[1, 0]],
                               [p[3, 0], p[2, 0], -p[1, 0], p[0, 0]]])
            return H_minus
        elif isinstance(p, cs.MX):
            H_minus = cs.vertcat(cs.horzcat(p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]),
                                cs.horzcat(p[1, 0], p[0, 0], p[3, 0], -p[2, 0]),
                                cs.horzcat(p[2, 0], -p[3, 0], p[0, 0], p[1, 0]),
                                cs.horzcat(p[3, 0], p[2, 0], -p[1, 0], p[0, 0]))
            return H_minus
        elif isinstance(p, cs.SX):
            H_minus = cs.vertcat(cs.horzcat(p[0, 0], -p[1, 0], -p[2, 0], -p[3, 0]),
                                cs.horzcat(p[1, 0], p[0, 0], p[3, 0], -p[2, 0]),
                                cs.horzcat(p[2, 0], -p[3, 0], p[0, 0], p[1, 0]),
                                cs.horzcat(p[3, 0], p[2, 0], -p[1, 0], p[0, 0]))
            return H_minus
        else:
            raise TypeError("Internal problem with the definition of the Quaternion, it should be a np.array, cs.MX or cs.SX.")