import torch
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.expressions.expression import Expression


class ElementwiseDotProduct(MulExpression):
    """
    This is an element wise dot product class. It should be used between two matrices with the same
    dimensions. It multiplies each row of the left hand side with the corresponding row of the
    right hand side.

    Parameters:
        lhs: expression
            The left hand side of the multiplication.
        rhs: expression
            The right hand side of the multiplication.
    """

    def numeric(self, values):
        """
        Elementwise dot product
        """
        raise NotImplementedError("Numeric for ElementwiseDotProduct should not be used.")
    
    def torch_numeric(self, values):
        return elementwise_dotproduct(values[0], values[1])
    
    @staticmethod
    def matmul_to_elementwise_dotproduct(expr: Expression) -> Expression:
        """
        This method returns a new expression where matrix multiplications are replaced with
        elementwise dot products if the args are vectors.
        """

        def _should_transform(expr: Expression):
            """
            This is a helper function that checks if this expression needs to be transformed
            """

            #Transform only multiplicaiton between two elements
            if len(expr.args) != 2:
                return False
            
            #Transform only if both are vectors
            for arg in expr.args:
                if len(arg.shape) != 1:
                    return False
            
            #Check if dot-product
            op_name = getattr(expr, "OP_NAME", None)
            return (op_name=="@") and (type(expr) != multiply)

        #Recursiv%ely change all the args of this expression
        args = [ElementwiseDotProduct.matmul_to_elementwise_dotproduct(arg) for arg in expr.args]
        expr.args = args
        #Change this expression if necessary
        if not _should_transform(expr):
            return expr
        return ElementwiseDotProduct(*expr.args)


def elementwise_dotproduct(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the dot product between the rows of a and the corresponding rows of b.
    
    Parameters:
        a, b (torch.Tensor):
            Input tensors
    
    Returns:
        res (torch.Tensor):
            A torch tensor with the same dimensions as a and b.
    
    Raises:
        ValueError if a and b are matrices not of the same size.
    """

    #If both are matrices and the dimensions don't match - raise an error
    if len(a.size())==2 and len(b.size())==2 and (a.size() != b.size()): 
        raise ValueError("Matrix dimensions mismatch: Elementwise multiplication is allowed for "
                         "matricres of the same dimensions only.")
    
    #If scalars or vectors, regular matmul
    if len(a.shape)<=1 or len(b.shape)<=1:
        return torch.matmul(a, b)

    batch_size = a.shape[0]
    row_len = a.shape[1]
    return torch.matmul(a.view(batch_size,1,row_len), b.view(batch_size,row_len,1)).reshape(-1)