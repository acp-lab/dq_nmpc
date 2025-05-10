import dataclasses
from typing import List
import casadi as cs
from dq_nmpc.type import Scalar, Vector
import numpy as np
from numbers import Number
from dq_nmpc import Quaternion
from dataclasses import field

@dataclasses.dataclass
class DualQuaternion():
     # This is to enable custom logic in the class's constructor for these fields.
    Qr: Quaternion = field(init=False, default=None)  # Real part of the DualQuaternion, not initialized by default.
    Qd: Quaternion = field(init=False, default=None)  # Dual part of the DualQuaternion, not initialized by default.
    
    def __init__(self, q_real = None, q_dual = None):
        """
        Initializes a DualQuaternion object with provided real (q_real) and dual (q_dual) quaternions.
        
        Parameters:
        - q_real (Quaternion, optional): The real part of the DualQuaternion. Must be an instance of the Quaternion class.
        - q_dual (Quaternion, optional): The dual part of the DualQuaternion. Must be an instance of the Quaternion class.

        Raises:
        - ValueError: If either q_real or q_dual is not provided, or if they are not both instances of Quaternion,
          or if they are not of the same numerical type (either both np.ndarray, cs.MX, or cs.SX).
        """
        if q_real is not None and q_dual is not None:
             # Validate that both q_real and q_dual are instances of Quaternion.
            if not all(isinstance(i, Quaternion) for i in [q_real, q_dual]):
                raise ValueError("Elements of the DualQuaternions must be instances of the Quaternion class.")
             # Ensure that the numerical types of the quaternions' values match among np.ndarray, cs.MX, or cs.SX.
            elif (isinstance(q_real.get, np.ndarray) and isinstance(q_dual.get, np.ndarray)) or (isinstance(q_real.get, cs.MX) and isinstance(q_dual.get, cs.MX)) or (isinstance(q_real.get, cs.SX) and isinstance(q_dual.get, cs.SX)):
                self.Qr = q_real # Assign the real part of the DualQuaternion.
                self.Qd = q_dual # Assign the dual part of the DualQuaternion.
            else:
                # Raises a ValueError if the quaternions' types do not match.
                raise ValueError("Elements of the DualQuaternions should be Quaternions of the same type: np.array, cs.MX, or cs.SX.")
        else:
            # Raises a ValueError indicating that both the real and dual parts are required.
            raise ValueError("Both the primal and dual quaternions must be provided.")

    def __str__(self) -> str:
        """
        Custom __str__ method for the DualQuaternion class.
        This method returns a string representation of the DualQuaternion object.
        """
        return f'DualQuaternion(Real: {self.Qr}, Dual: {self.Qd})'

    def __getattr__(self, name):
        """
        Overrides the default attribute access method for the DualQuaternion class.
        This method attempts to delegate attribute access first to the real part (Qr) and then to the dual part (Qd) of the DualQuaternion.
        If the attribute is not found in either part, an AttributeError is raised, indicating the absence of the requested attribute in the DualQuaternion object.

        Returns:
        - The value of the attribute from either the real or dual part, if found.

        Raises:
        - AttributeError: If the attribute is not found in both the real and dual parts of the DualQuaternion.
        """
        try:
            return getattr(self.Qr, name)
        except AttributeError:
            pass  # If attribute is not found in Qr, try Qd

        try:
            return getattr(self.Qd, name)
        except AttributeError:
            # If attribute is not found in both, raise an AttributeError
            raise AttributeError(f"'DualQuaternion' object has no attribute '{name}'")

    @staticmethod
    def from_pose(quat: Vector, trans: Vector) -> "DualQuaternion":
        """
        Creates a DualQuaternion from a given pose of a rigid body, defined by a quaternion for orientation and a vector for translation.

        Parameters:
        - quat (Vector): The orientation of the rigid body represented as a quaternion.
        - trans (Vector): The translation of the rigid body represented as a vector.

        Returns:
        - DualQuaternion: An instance of DualQuaternion representing the pose of the rigid body.
        """
         # Convert the translation vector into a Quaternion
        t = Quaternion(q= trans)
        # Convert the orientation quaternion into a Quaternion
        q = Quaternion(q = quat)
        q_r = q
        # Calculate the dual part using the formula: (1/2) * (quaternion * translation quaternion).
        # This represents the translation component of the DualQuaternion.
        # Dual Part (1/2) * t * q   
        # Dual Part (1/2) * q * t
        #q_d = (1/2)* (q*t)
        q_d = (1/2)* (t*q)
        return DualQuaternion(q_real = q_r, q_dual = q_d)

    def __mul__(self, q2: "DualQuaternion") -> "DualQuaternion":
        """
        Defines the left multiplication operation for DualQuaternions with another DualQuaternion or a scalar value.

        Parameters:
        - q2 (DualQuaternion, Number, cs.MX, or cs.SX): The multiplicand, which can be another DualQuaternion or a scalar value (Number, cs.MX, or cs.SX).

        Returns:
        - DualQuaternion: The result of the multiplication, which should be a DualQuaternion.
        """
        if isinstance(q2, DualQuaternion):
             # If q2 is a DualQuaternion, return the product of self and q2 using the defined product method.
            return DualQuaternion.product(self, q2)
        elif (isinstance(q2, Number)):
            # Scalar multiplication with a Number, applicable when the Dualquaternion's numerical type is np.ndarray.
            q1r = self.Qr
            q1d = self.Qd
            qr_out =  q1r * q2
            qd_out =  q1d * q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.MX) and isinstance(self.Qr.get, cs.MX)):
            if q2.shape[0]==1:
            # Handle multiplication with cs.MX types or scalar multiplication when the Dualquaternion's type is cs.MX.
                q1r = self.Qr
                q1d = self.Qd
                qr_out =  q1r * q2
                qd_out =  q1d * q2
                return DualQuaternion(q_real = qr_out, q_dual = qd_out)
            else:
                raise TypeError("Left Multiplication with DualQuaternion is only defined for DualQuaternions and scalars of the same type.")

        elif(isinstance(q2, cs.SX) and isinstance(self.Qr.get, cs.SX)):
            if q2.shape[0]==1:
                # Handle multiplication with cs.SX types or scalar multiplication when the quaternion's type is cs.SX.
                q1r = self.Qr
                q1d = self.Qd
                qr_out =  q1r * q2
                qd_out =  q1d * q2
                return DualQuaternion(q_real = qr_out, q_dual = qd_out)
            else:
                raise TypeError("Left Multiplication with DualQuaternion is only defined for DualQuaternions and scalars of the same type.")
        else:
            raise TypeError("Left Multiplication with DualQuaternion is only defined for DualQuaternions and scalars of the same type.")

    def __rmul__(self, q2) -> "DualQuaternion":
        """
        Defines the right multiplication operation for DualQuaternions with a scalar or compatible numeric type.

        This method enables scalar multiplication from the right-hand side, allowing expressions like scalar * DualQuaternion. 

        Parameters:
        - q2: The multiplicand, which can be a scalar (Number) or a compatible numeric type (cs.MX or cs.SX).

        Returns:
        - DualQuaternion: The result of the multiplication, a new DualQuaternion instance with both real and dual parts multiplied by q2.
        """
        if (isinstance(q2, Number)):
            # Scalar multiplication with a Number when the quaternion's type is np.ndarray.
            q1r = self.Qr
            q1d = self.Qd
            qr_out =  q2 * q1r
            qd_out =  q2 * q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)

        elif(isinstance(q2, cs.MX) and isinstance(self.Qr.get, cs.MX)):
            if q2.shape[0]==1:
                # Handles multiplication with cs.MX types or scalar multiplication when the quaternion's type is cs.MX.
                q1r = self.Qr
                q1d = self.Qd
                qr_out =  q2 * q1r
                qd_out =  q2 * q1d
                return DualQuaternion(q_real = qr_out, q_dual = qd_out)
            else:
                raise TypeError("Right Multiplication with DualQuaternion is only defined for scalars of the same type.")

        elif(isinstance(q2, cs.SX) and isinstance(self.Qr.get, cs.SX)):
            # Handles multiplication with cs.SX types or scalar multiplication when the quaternion's type is cs.SX.
            if q2.shape[0]==1:
                q1r = self.Qr
                q1d = self.Qd
                qr_out =  q2 * q1r
                qd_out =  q2 * q1d
                return DualQuaternion(q_real = qr_out, q_dual = qd_out)
            else:
                raise TypeError("Right Multiplication with DualQuaternion is only defined for scalars of the same type.")

        else:
            raise TypeError("Right Multiplication with DualQuaternion is only defined for scalars of the same type.")

    @staticmethod
    def product(p: "DualQuaternion", q: "DualQuaternion") -> "DualQuaternion":
        """
        Computes the Hamilton product of two DualQuaternions.

        This method calculates the product of two DualQuaternions, `p` and `q`, 
        following the rules of dual quaternion multiplication. The method ensures that 
        the numerical types of the quaternions (np.ndarray, cs.MX, or cs.SX) are compatible 
        before performing the multiplication.

        Parameters:
        - p (DualQuaternion): The first DualQuaternion in the multiplication.
        - q (DualQuaternion): The second DualQuaternion in the multiplication.

        Returns:
        - DualQuaternion: The product of `p` and `q` as a new DualQuaternion.

        Raises:
        - TypeError: If the elements of `p` and `q` are not of the same numerical type.
        """
        # Extract the real and dual parts of both DualQuaternions.
        q1r = p.Qr
        q1d = p.Qd

        q2r = q.Qr
        q2d = q.Qd
        # Check if the numerical types of the real parts of both quaternions are np.ndarray.
        if isinstance(q1r.get, np.ndarray) and isinstance(q2r.get, np.ndarray):  # Use Vector directly without parentheses
            real = q1r * q2r
            dual = q1r * q2d + q1d * q2r

        # Check if the numerical types of the real parts are cs.MX.
        elif isinstance(q1r.get, cs.MX) and isinstance(q2r.get, cs.MX):
            real = q1r * q2r
            dual = q1r * q2d + q1d * q2r

        # Check if the numerical types of the real parts are cs.SX.
        elif isinstance(q1r.get, cs.SX) and isinstance(q2r.get, cs.SX):
            real = q1r * q2r
            dual = q1r * q2d + q1d * q2r
        else:
            raise TypeError("The elements of both DualQuaternions should be of the same type.")
        return DualQuaternion(q_real= real, q_dual= dual)

    @property
    def get(self) -> Vector:
        """
        Retrieves a combined vector representation of the real and dual parts of the DualQuaternion.

        This property concatenates the real part (Qr) and the dual part (Qd) of the DualQuaternion into a single vector.
        The method ensures compatibility with the numerical types (np.ndarray, cs.MX, or cs.SX) of the quaternion components.

        Returns:
        - Vector: A combined vector representation of the DualQuaternion, stacking the real part on top of the dual part.

        Raises:
        - TypeError: If the numerical types of Qr and Qd do not match or are not supported (not np.ndarray, cs.MX, or cs.SX).
        """
        # Retrieve the vector representation of the real and dual parts.
        Qr_aux = self.Qr.get
        Qd_aux = self.Qd.get

        # Check if the numerical types of both parts are np.ndarray and concatenate vertically.
        if isinstance(Qr_aux, np.ndarray) and (Qd_aux, np.array):  # Use Vector directly without parentheses
            return np.vstack([self.Qr.get, self.Qd.get])
        # Check if the numerical types are cs.MX and concatenate vertically.
        elif isinstance(Qr_aux, cs.MX) and (Qd_aux, cs.MX):
            return cs.vertcat(self.Qr.get, self.Qd.get)
        # Check if the numerical types are cs.SX and concatenate vertically.
        elif isinstance(Qr_aux, cs.SX) and (Qd_aux, cs.SX):
            return cs.vertcat(self.Qr.get, self.Qd.get)
        else:
            raise TypeError("Internal problem with the definition of the DualQuaternion; it should be of the same type: np.array, cs.MX, or cs.SX.")

    def conjugate(self) -> "DualQuaternion":
        """
        Computes the conjugate of the DualQuaternion.

        The conjugate of a DualQuaternion involves taking the conjugate of both its real (Qr) and dual (Qd) parts. 
        
        Returns:
        - DualQuaternion: A new DualQuaternion instance where both the real and dual parts are conjugated.
        """
        # Access the real and dual parts of the DualQuaternion.
        qr = self.Qr
        qd = self.Qd
        # Compute the conjugate of the real part.
        qr_conjugate = qr.conjugate()
        # Compute the conjugate of the dual part.
        qd_conjugate = qd.conjugate()
        return DualQuaternion(q_real= qr_conjugate, q_dual = qd_conjugate)

    @property
    def norm_dual_control(self):
        """
        Calculates the norms of the real and dual parts of the DualQuaternion.

        Returns:
        - Tuple[Scalar, Scalar]: A tuple containing the norm of the real part and the calculated value for the dual part
        """
         # Access the real and dual parts of the DualQuaternion.
        qr = self.Qr
        qd = self.Qd

        if isinstance(qr.get, np.ndarray) and isinstance(qd.get, np.ndarray):  # Use Vector directly without parentheses
            real_norm = qr.norm
            # Calculate the dual norm by dividing the auxiliary value by the real part's norm.
            dual_norm = qd.norm
            #dual_norm = (dual_norm_aux[0, 0])
            #dual_norm = qd.norm
            #print(dual_norm)
            return real_norm, dual_norm

        elif isinstance(qr.get, cs.MX) and isinstance(qd.get, cs.MX):
            real_norm = qr.norm
            # Calculate the dual norm by dividing the auxiliary value by the real part's norm.
            dual_norm = qd.norm
            #dual_norm = (dual_norm_aux[0, 0])
            #dual_norm = qd.norm
            #print(dual_norm)
            return real_norm, dual_norm

        elif isinstance(qr.get, cs.SX) and isinstance(qd.get, cs.SX):
            real_norm = qr.norm
            # Calculate the dual norm by dividing the auxiliary value by the real part's norm.
            dual_norm = qd.norm
            #dual_norm = (dual_norm_aux[0, 0])
            #dual_norm = qd.norm
            #print(dual_norm)
            return real_norm, dual_norm
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
    @property
    def norm_dual(self):
        """
        Calculates the norms of the real and dual parts of the DualQuaternion.

        Returns:
        - Tuple[Scalar, Scalar]: A tuple containing the norm of the real part and the calculated value for the dual part
        """
         # Access the real and dual parts of the DualQuaternion.
        qr = self.Qr
        qd = self.Qd

        if isinstance(qr.get, np.ndarray) and isinstance(qd.get, np.ndarray):  # Use Vector directly without parentheses
            real_norm = qr.square_norm
            # Compute an auxiliary value for the dual norm.
            dual_norm_aux = 2*(qr.get.T@qd.get)/1
            # Calculate the dual norm by dividing the auxiliary value by the real part's norm.
            dual_norm = dual_norm_aux[0, 0]
            #dual_norm = (dual_norm_aux[0, 0])
            #dual_norm = qd.norm
            #print(dual_norm)
            return real_norm, dual_norm

        elif isinstance(qr.get, cs.MX) and isinstance(qd.get, cs.MX):
            # Calculate the norm of the real part directly.
            real_norm = qr.square_norm
            # Compute an auxiliary value for the dual norm.
            dual_norm_aux = 2*(qr.get.T@qd.get)/1
            # Calculate the dual norm by dividing the auxiliary value by the real part's norm.
            dual_norm = dual_norm_aux[0, 0]
            #dual_norm = (dual_norm_aux[0, 0])
            #dual_norm = qd.norm
            #print(dual_norm)
            return real_norm, dual_norm

            return real_norm, dual_norm
        elif isinstance(qr.get, cs.SX) and isinstance(qd.get, cs.SX):
            # Calculate the norm of the real part directly.
            real_norm = qr.square_norm
            # Compute an auxiliary value for the dual norm.
            dual_norm_aux = 2*(qr.get.T@qd.get)/1
            # Calculate the dual norm by dividing the auxiliary value by the real part's norm.
            dual_norm = dual_norm_aux[0, 0]
            #dual_norm = (dual_norm_aux[0, 0])
            #dual_norm = qd.norm
            #print(dual_norm)
            return real_norm, dual_norm
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
    
    @property
    def get_trans(self) -> "Quaternion":
        """
        Extracts the translation component from the DualQuaternion.

        This property calculates the translation vector 't' encoded within the dual part of the DualQuaternion
        
        Returns:
        - Quaternion: A Quaternion object representing the translation vector 't'.
        """
        # Check this section, this is related to the definition of a rigid body trasnformation
        # q = qr + e(1/2)*t*qr                   - t = 2 * qd * qr_c
        # q = qr + e(1/2)*qr*t                   - t = 2 * qr_c * qd
        # Access the real part (qr) and its conjugate (qr_c)
        qr = self.Qr
        qr_c = qr.conjugate()

        # Access the dual part (qd).
        qd = self.Qd
        # Calculate the translation component 't' using the formula 't = 2 * qr_c * qd.
        # This is based on the dual quaternion representation of rigid body transformations.
        #t = 2*qr_c * qd
        t = 2* qd * qr_c

        # Retrieve the data of 't' to construct a new Quaternion representing the translation.
        t_data = t.get
        return Quaternion(q = t_data)

    @property
    def get_quat(self) -> "Quaternion":
        """
         This property accesses the real part of the DualQuaternion, 'qr', which encodes the rotation 

        Returns:
        - Quaternion: A Quaternion object representing the rotation encoded in the DualQuaternion.
        """
        # Access the real part (qr) of the DualQuaternion, which represents the rotation.
        qr = self.Qr
        # Retrieve the quaternion data from 'qr'.
        qr_data = qr.get
        return Quaternion(q = qr_data)

    @property
    def get_real(self) -> "Quaternion":
        """
        Retrieves the real part of the DualQuaternion as a Quaternion object.

        Returns:
        - Quaternion: A new Quaternion object instantiated with the data from the real part of the DualQuaternion.
        """
        # Access the real part ('qr') of the DualQuaternion.
        qr = self.Qr
        # Retrieve the underlying data from 'qr'.
        qr_data = qr.get
        return Quaternion(q = qr_data)

    @property
    def get_dual(self) -> "Quaternion":
        """
        Retrieves the dual part of the DualQuaternion as a Quaternion object.

        Returns:
        - Quaternion: A new Quaternion object instantiated with the data from the dual part of the DualQuaternion.
        """
        # Access the dual part ('qd') of the DualQuaternion.
        qd = self.Qd
        # Retrieve the underlying data from 'qd'.
        qd_data = qd.get
        return Quaternion(q = qd_data)

    def __add__(self, q2: "DualQuaternion") -> "DualQuaternion":
        if isinstance(q2, DualQuaternion):
            return DualQuaternion.add(self, q2)
        elif (isinstance(q2, Number)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q1r + q2
            qd_out = q1d + q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.MX) and isinstance(self.Qr.get, cs.MX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q1r + q2
            qd_out = q1d + q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.SX) and isinstance(self.Qr.get, cs.SX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q1r + q2
            qd_out = q1d + q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        else:
            raise TypeError("Right addition is only defined for DualQuaternions and scalars.")

    def __radd__(self, q2: "DualQuaternion") -> "DualQuaternion":
        if isinstance(q2, DualQuaternion):
            return DualQuaternion.add(q2, self)
        elif (isinstance(q2, Number)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q2 + q1r
            qd_out = q2 + q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.MX) and isinstance(self.Qr.get, cs.MX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q2 + q1r
            qd_out = q2 + q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.SX) and isinstance(self.Qr.get, cs.SX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q2 + q1r
            qd_out = q2 + q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        else:
            raise TypeError("Left addtion is only defined for DualQuaternions and scalars")

    @staticmethod
    def add(p: "DualQuaternion", q: "DualQuaternion") -> "DualQuaternion":
        # Get elements of the dual quaternions
        q1r = p.Qr
        q1d = p.Qd

        q2r = q.Qr
        q2d = q.Qd

        if isinstance(q1r.get, np.ndarray) and isinstance(q2r.get, np.ndarray):  # Use Vector directly without parentheses
            real = q1r + q2r
            dual = q1d + q2d

        elif isinstance(q1r.get, cs.MX) and isinstance(q2r.get, cs.MX):
            real = q1r + q2r
            dual = q1d + q2d

        elif isinstance(q1r.get, cs.SX) and isinstance(q2r.get, cs.SX):
            real = q1r + q2r
            dual = q1d + q2d
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
        return DualQuaternion(q_real= real, q_dual= dual)

    def __sub__(self, q2: "DualQuaternion") -> "DualQuaternion":
        if isinstance(q2, DualQuaternion):
            return  DualQuaternion.sub(self, q2)
        elif (isinstance(q2, Number)) :
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q1r - q2
            qd_out = q1d - q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.MX) and isinstance(self.Qr.get, cs.MX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q1r - q2
            qd_out = q1d - q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.SX) and isinstance(self.Qr.get, cs.SX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q1r - q2
            qd_out = q1d - q2
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        else:
            raise TypeError("Right substracion only defined for DualQuaternions and scalars")

    def __rsub__(self, q2: "DualQuaternion") -> "DualQuaternion":
        if isinstance(q2, Quaternion):
            return  DualQuaternion.sub(q2, self)
        elif (isinstance(q2, Number)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q2 - q1r
            qd_out = q2 - q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.MX) and isinstance(self.Qr.get, cs.MX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q2 - q1r
            qd_out = q2 - q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        elif(isinstance(q2, cs.SX) and isinstance(self.Qr.get, cs.SX)):
            q1r = self.Qr
            q1d = self.Qd
            qr_out = q2 - q1r
            qd_out = q2 - q1d
            return DualQuaternion(q_real = qr_out, q_dual = qd_out)
        else:
            raise TypeError("Left Substraction is only defined for DualQuaternions and scalars")

    @staticmethod
    def sub(p: "DualQuaternion", q: "DualQuaternion") -> "DualQuaternion":
        # Get elements of the dual quaternions
        q1r = p.Qr
        q1d = p.Qd

        q2r = q.Qr
        q2d = q.Qd

        if isinstance(q1r.get, np.ndarray) and isinstance(q2r.get, np.ndarray):  # Use Vector directly without parentheses
            real = q1r - q2r
            dual = q1d - q2d

        elif isinstance(q1r.get, cs.MX) and isinstance(q2r.get, cs.MX):
            real = q1r - q2r
            dual = q1d - q2d

        elif isinstance(q1r.get, cs.SX) and isinstance(q2r.get, cs.SX):
            real = q1r - q2r
            dual = q1d - q2d
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
        return DualQuaternion(q_real= real, q_dual= dual)

    def set(self, q_real = None, q_dual = None):
        # Set new elements of the Dualquaternion elements
        if q_real is not None and q_dual is not None:
            if not all(isinstance(i, Quaternion) for i in [q_real, q_dual]):
                raise ValueError("Elements of the DualQuaternions should be Quaternions")
            elif (isinstance(q_real.get, np.ndarray) and isinstance(q_dual.get, np.ndarray)) or (isinstance(q_real.get, cs.MX) and isinstance(q_dual.get, cs.MX)) or (isinstance(q_real.get, cs.SX) and isinstance(q_dual.get, cs.SX)):
                self.Qr = q_real
                self.Qd = q_dual
            else:
                raise ValueError("Elements of the DualQuaternions should be Quaternions of the same type: np.array. cs.MX or c.SX")
        else:
            # Handle case where q_primal or q_dual are not provided
            raise ValueError("Both primal and dual quaternions must be provided")

    def ln_control(self):
        # Log mapping of the dualQuaternion
        q1r = self.Qr
        q1d = self.Qd
        # Get the log mapping of the quaterion inside the DualQuaternion
        dual_axis = self.axis_angle_dual()

        # Get real and dual values
        axis_angle_real = dual_axis.get_real.get
        axis_angle_dual = dual_axis.get_dual.get

        if isinstance(q1r.get, np.ndarray) and isinstance(q1d.get, np.ndarray):  # Use Vector directly without parentheses
            # Get translation of the dual quaternion
            theta = axis_angle_real[0, 0]
            n = axis_angle_real[1:4, 0]

            p = axis_angle_dual[0, 0]
            trans = axis_angle_dual[1:4, 0]

            # Dual Part
            s = trans/p
            ps = (1/2) * trans
            result_ps = np.vstack((0.0, ps[0], ps[1], ps[2]))

            # Real Part
            ntheta = (1/2)*theta * n
            result_ntheta = np.vstack((0.0, ntheta[0], ntheta[1], ntheta[2]))


        elif isinstance(q1r.get, cs.MX) and isinstance(q1d.get, cs.MX):
            # Get translation of the dual quaternion
            theta = axis_angle_real[0, 0]
            n = axis_angle_real[1:4, 0]

            p = axis_angle_dual[0, 0]
            trans = axis_angle_dual[1:4, 0]

            # Dual Part
            s = trans/p
            ps = (1/2) * trans
            result_ps = cs.vertcat(0.0, ps[0, 0], ps[1, 0], ps[2, 0])

            # Real Part
            ntheta = (1/2)*theta * n
            result_ntheta = cs.vertcat(0.0, ntheta[0, 0], ntheta[1, 0], ntheta[2, 0])
        elif isinstance(q1r.get, cs.SX) and isinstance(q1d.get, cs.SX):
            # Get translation of the dual quaternion
            theta = axis_angle_real[0, 0]
            n = axis_angle_real[1:4, 0]

            p = axis_angle_dual[0, 0]
            trans = axis_angle_dual[1:4, 0]

            # Dual Part
            s = trans/p
            ps = (1/2) * trans
            result_ps = cs.vertcat(0.0, ps[0, 0], ps[1, 0], ps[2, 0])

            # Real Part
            ntheta = (1/2)* theta * n
            result_ntheta = cs.vertcat(0.0, ntheta[0, 0], ntheta[1, 0], ntheta[2, 0])
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
        Dual_ln = DualQuaternion(q_real= Quaternion(q = result_ntheta), q_dual= Quaternion(q = result_ps))
        return Dual_ln

    def ln_dual(self):
        # Log mapping of the dualQuaternion
        q1r = self.Qr
        q1d = self.Qd
        # Get the log mapping of the quaterion inside the DualQuaternion
        dual_axis = self.axis_angle_dual()

        # Get real and dual values
        axis_angle_real = dual_axis.get_real.get
        axis_angle_dual = dual_axis.get_dual.get

        if isinstance(q1r.get, np.ndarray) and isinstance(q1d.get, np.ndarray):  # Use Vector directly without parentheses
            # Get translation of the dual quaternion
            theta = axis_angle_real[0, 0]
            n = axis_angle_real[1:4, 0]

            p = axis_angle_dual[0, 0]
            trans = axis_angle_dual[1:4, 0]

            # Dual Part
            s = trans/p
            ps = (1/2) * trans
            result_ps = np.vstack((0.0, ps[0], ps[1], ps[2]))

            # Real Part
            ntheta = (1/2)*theta * n
            result_ntheta = np.vstack((0.0, ntheta[0], ntheta[1], ntheta[2]))


        elif isinstance(q1r.get, cs.MX) and isinstance(q1d.get, cs.MX):
            # Get translation of the dual quaternion
            theta = axis_angle_real[0, 0]
            n = axis_angle_real[1:4, 0]

            p = axis_angle_dual[0, 0]
            trans = axis_angle_dual[1:4, 0]

            # Dual Part
            s = trans/p
            ps = (1/2) * trans
            result_ps = cs.vertcat(0.0, ps[0, 0], ps[1, 0], ps[2, 0])

            # Real Part
            ntheta = (1/2)*theta * n
            result_ntheta = cs.vertcat(0.0, ntheta[0, 0], ntheta[1, 0], ntheta[2, 0])
        elif isinstance(q1r.get, cs.SX) and isinstance(q1d.get, cs.SX):
            # Get translation of the dual quaternion
            theta = axis_angle_real[0, 0]
            n = axis_angle_real[1:4, 0]

            p = axis_angle_dual[0, 0]
            trans = axis_angle_dual[1:4, 0]

            # Dual Part
            s = trans/p
            ps = (1/2) * trans
            result_ps = cs.vertcat(0.0, ps[0, 0], ps[1, 0], ps[2, 0])

            # Real Part
            ntheta = (1/2)* theta * n
            result_ntheta = cs.vertcat(0.0, ntheta[0, 0], ntheta[1, 0], ntheta[2, 0])
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
        Dual_ln = DualQuaternion.from_pose(quat= result_ntheta, trans= result_ps)
        return Dual_ln

    def axis_angle_dual(self)->"DualQuaternion":
        # Log mapping of the dualQuaternion
        q1r = self.Qr
        q1d = self.Qd

        if isinstance(q1r.get, np.ndarray) and isinstance(q1d.get, np.ndarray):  # Use Vector directly without parentheses
            q1r_angle_axis = q1r.angle_axis()
            # Get translation of the dual quaternion
            trans = self.get_trans

            p = trans.norm

            result = np.vstack((p, trans.get[1, 0], trans.get[2, 0], trans.get[3, 0]))

            Dual_ln = DualQuaternion(q_real=Quaternion(q = q1r_angle_axis), q_dual= Quaternion(q = result))

        elif isinstance(q1r.get, cs.MX) and isinstance(q1d.get, cs.MX):
            # Get the log mapping of the quaterion inside the DualQuaternion
            q1r_angle_axis = q1r.angle_axis()
            # Get translation of the dual quaternion
            trans = self.get_trans

            p = trans.norm

            result = cs.vertcat(p, trans.get[1, 0], trans.get[2, 0], trans.get[3, 0])

            Dual_ln = DualQuaternion(q_real=Quaternion(q = q1r_angle_axis), q_dual= Quaternion(q = result))
        elif isinstance(q1r.get, cs.SX) and isinstance(q1d.get, cs.SX):
            # Get the log mapping of the quaterion inside the DualQuaternion
            q1r_angle_axis = q1r.angle_axis()
            # Get translation of the dual quaternion
            trans = self.get_trans

            p = trans.norm

            result = cs.vertcat(p, trans.get[1, 0], trans.get[2, 0], trans.get[3, 0])

            Dual_ln = DualQuaternion(q_real=Quaternion(q = q1r_angle_axis), q_dual= Quaternion(q = result))
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
        return Dual_ln

    @staticmethod
    def vector_dot(p: "DualQuaternion", q: "DualQuaternion") -> "DualQuaternion":
        # Get elements of the dual quaternions
        q1r = p.Qr
        q1d = p.Qd

        q2r = q.Qr
        q2d = q.Qd
        # Check for the type of variable of both elements
        if isinstance(q1r.get, np.ndarray) and isinstance(q2r.get, np.ndarray):  # Use Vector directly without parentheses
            real = q1r.get * q2r.get
            dual = q1d.get * q2d.get

        elif isinstance(q1r.get, cs.MX) and isinstance(q2r.get, cs.MX):
            real = q1r.get * q2r.get
            dual = q1d.get * q2d.get

        elif isinstance(q1r.get, cs.SX) and isinstance(q2r.get, cs.SX):
            real = q1r.get * q2r.get
            dual = q1d.get * q2d.get
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")

        # Creation of the quaternions
        real_quat = Quaternion(q = real)
        dual_quat = Quaternion(q = dual)

        return DualQuaternion(q_real= real_quat, q_dual= dual_quat)

    def vector_dot_product(self, q2):
        if isinstance(q2, DualQuaternion):
            product = DualQuaternion.vector_dot(self, q2)
            return product
        else:
            raise TypeError("Vector Dot Product only defined for DualQuaternions")

    @property
    def H_plus_dual(self):
        # Log mapping of the dualQuaternion
        q1r = self.Qr
        q1d = self.Qd
        # Get the log mapping of the quaterion inside the DualQuaternion
        H_r_plus = q1r.H_plus
        H_d_plus = q1d.H_plus

        if isinstance(q1r.get, np.ndarray) and isinstance(q1d.get, np.ndarray):  # Use Vector directly without parentheses
            zeros = np.zeros((4, 4))
            Hplus = np.block([[H_r_plus, zeros],
                               [H_d_plus, H_r_plus]])
            # Get translation of the dual quaternion
        elif isinstance(q1r.get, cs.MX) and isinstance(q1d.get, cs.MX):
            # Get translation of the dual quaternion
            zeros = cs.MX.zeros(4, 4)
            Hplus = cs.vertcat(cs.horzcat(H_r_plus, zeros),
                        cs.horzcat(H_d_plus, H_r_plus))
        elif isinstance(q1r.get, cs.SX) and isinstance(q1d.get, cs.SX):
            # Get translation of the dual quaternion
            zeros = cs.SX.zeros(4, 4)
            Hplus = cs.vertcat(cs.horzcat(H_r_plus, zeros),
                        cs.horzcat(H_d_plus, H_r_plus))
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")

        return Hplus

    @property
    def H_minus_dual(self):
        # Log mapping of the dualQuaternion
        q1r = self.Qr
        q1d = self.Qd
        # Get the log mapping of the quaterion inside the DualQuaternion
        H_r_minus = q1r.H_minus
        H_d_minus = q1d.H_minus

        if isinstance(q1r.get, np.ndarray) and isinstance(q1d.get, np.ndarray):  # Use Vector directly without parentheses
            zeros = np.zeros((4, 4))
            Hminus = np.block([[H_r_minus, zeros],
                               [H_d_minus, H_r_minus]])
            # Get translation of the dual quaternion
        elif isinstance(q1r.get, cs.MX) and isinstance(q1d.get, cs.MX):
            # Get translation of the dual quaternion
            zeros = cs.MX.zeros(4, 4)
            Hminus = cs.vertcat(cs.horzcat(H_r_minus, zeros),
                        cs.horzcat(H_d_minus, H_r_minus))
        elif isinstance(q1r.get, cs.SX) and isinstance(q1d.get, cs.SX):
            # Get translation of the dual quaternion
            zeros = cs.SX.zeros(4, 4)
            Hminus = cs.vertcat(cs.horzcat(H_r_minus, zeros),
                        cs.horzcat(H_d_minus, H_r_minus))
        else:
            raise TypeError("The elements of both Dualquaternions should be of the same type.")
        return Hminus