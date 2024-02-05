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

    def __init__(self, *args):
        super().__init__(*args)

    def numeric(self, values):
        """
        Elementwise dot product
        """
        raise NotImplementedError("Numeric for ElementwiseDotProduct should not be used.")
    
    def torch_numeric(self, values):
        return self.elementwise_dotproduct(values[0], values[1])
    
    def elementwise_dotproduct(self, *args) -> torch.Tensor:
        """
        This function calculates the dot product between the rows of a and the
        corresponding rows of b.
        This function can handle any combination of matrix/vector @ matrix/vector,
        where each element might be batched.

        Parameters:
            *args:
                Input tensors. Should be exactly two.
        
        Returns:
            res (torch.Tensor):
                A torch tensor with the same dimensions as a and b.
        
        Raises:
            ValueError if a and b are matrices not of the same size, or if not 2 tensors are passed.
        """
        def _get_is_batch_vec(a: torch.Tensor, expr_arg_shape: tuple[int]) -> int | None:
            """
            This helper function returns the number of batches for a tensor, or None if it is not
            batched. The batch size is always assumed to be the 1st dimension.
            
            Raise:
                ValueError if the shape of a does not match
            """
            is_batch = len(a.shape)-len(expr_arg_shape) == 1
            #If batch, the 0-th dim is the batch size
            a_true_shape = a.shape[1:] if is_batch else a.shape
            #Convert both to lists to handle torch.Size, np.size, etc.
            if list(a_true_shape) != list(expr_arg_shape):
                raise ValueError(f"Dimensions mismatch: got {a.shape} and {expr_arg_shape}.")
            return a.shape[0] if is_batch else None
        
        def _get_batch_size(is_batch_vec: list) -> int:
            """
            This is a helper function that returns the batch size.
            
            Raise:
                ValueError if given multiple disagreeing batches
            """
            batch_size = None
            for batch in is_batch_vec:
                if batch is None:
                    continue
                if batch_size is None:
                    batch_size = batch
                elif batch_size != batch:
                    raise ValueError(f"Mismatch in batch size: Got {batch_size} and {batch}")
            return batch_size

        def _gen_batched_args(args: list, is_batch_vec: list) -> list:
            """
            This is a helper function that returns a new list of args that are stacked
            """
            batched_args = []
            for i, is_batch in enumerate(is_batch_vec):
                if is_batch is None:
                    batched_args.append(args[i].expand(size=([batch_size] + list(args[i].shape))))
                else:
                    batched_args.append(args[i])
            return batched_args
        
        def _gen_is_mat_vec(self) -> list[bool]:
            """
            This is a helper function that returns a boolean list of all the matrices indices
            """
            is_mat_vec = [None]*len(self.args)
            for i, arg in enumerate(self.args):
                is_mat_vec[i] = (len(arg.shape)>=2)
            return is_mat_vec
        
        def _proprocess_args(self, args: tuple) -> list:
            """
            This function preprocess args: Converts it to a list, and re-transposes the RHS if BOTH:
            1. It is a batched vector
            2. The LHS is a constant vector
            """
            def _is_batched_vec(self, args: tuple, ind: int) -> bool:
                """
                This is a helper function that returns True if args[i] is a batched vector.
                """
                return _is_vec(self.args[ind]) and (not _is_vec(args[ind]))

            args = list(args)
            is_vec_lhs = _is_vec(self.args[0])
            is_batched_vec_lhs = _is_batched_vec(self, args, 0)
            is_batched_vec_rhs = _is_batched_vec(self, args, 1)
            if is_vec_lhs and (not is_batched_vec_lhs) and is_batched_vec_rhs:
                args[1] = args[1].T
            return args
        
        def _is_vec(obj) -> bool:
            """
            This is a helper function that returns True if an object (tensor, CP.Variable, ...) is
            a vector, and False otherwise. The object needs to have a shape attribute.
            """
            return len(obj.shape)==1

        #Check if exactly 2 tensors are passed
        if len(args) != 2:
            raise ValueError(f"Expected 2 elements for @, but got {len(args)} instead.")

        #If vec-vec and have batch - need to transpose the items. The reason is that gen_torch_exp
        #in CVXPY will incorrectly transpose vector batches, thinking they are matrices.
        args = _proprocess_args(self, args)

        #See which dimensions are batched
        is_batch_vec = [None]*len(args)
        for i, arg in enumerate(args):
            is_batch_vec[i] = _get_is_batch_vec(arg, self.args[i].shape)

        batch_size = _get_batch_size(is_batch_vec)

        #If the dimensions match the saved dimensions - regular matmul
        if batch_size is None:
            return torch.matmul(*args)
        
        #Else: batched matmul
        #If one of the args is not batched - stack it
        batched_args = _gen_batched_args(args, is_batch_vec)
        
        #Check which argument is supposed to be a matrix
        is_mat_vec = _gen_is_mat_vec(self)

        #matrix-matrix
        if all(is_mat_vec):
            return torch.matmul(*batched_args)
        
        #vector-vector
        elif not any(is_mat_vec):
            row_len = batched_args[0].shape[1]
            return torch.matmul(batched_args[0].view(batch_size,1,row_len),
                                batched_args[1].view(batch_size,row_len,1)).reshape(-1)
                        
        #vector-matrix / matrix-vector
        else:
            #Get the indices of the vector and the matrix
            mat_ind = 0 if is_mat_vec[0] else 1
            vec_ind = 1-mat_ind

            #batched matrix, unbatched vector
            if (is_batch_vec[mat_ind]) and (not is_batch_vec[vec_ind]):
                if mat_ind==0:
                    return torch.matmul(batched_args[mat_ind], args[vec_ind])
                else:
                    return torch.matmul(args[vec_ind], batched_args[mat_ind])

            #batched matrix, batched vector
            elif (is_batch_vec[mat_ind]) and (is_batch_vec[vec_ind]):
                batch_size, row_len = args[vec_ind].shape
                if mat_ind==0:
                    return torch.matmul(batched_args[mat_ind],
                                        args[vec_ind].view(batch_size, row_len, 1)).squeeze(-1)
                else:
                    return torch.matmul(args[vec_ind].view(batch_size, 1, row_len),
                                        batched_args[mat_ind]).squeeze(1)

            #unbatched matrix, unbatched vector
            elif (not is_batch_vec[mat_ind]) and (not is_batch_vec[vec_ind]):
                return torch.matmul(*args)

            #unbatched matrix, batched vector
            elif (not is_batch_vec[mat_ind]) and (is_batch_vec[vec_ind]):
                if mat_ind==0:
                    return torch.matmul(batched_args[vec_ind], args[mat_ind].T)
                else:
                    return torch.matmul(batched_args[vec_ind], args[mat_ind])

        
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

            # #Transform only multiplicaiton between two elements
            # if len(expr.args) != 2:
            #     return False
            
            # #Transform only if both are vectors
            # for arg in expr.args:
            #     if len(arg.shape) != 1:
            #         return False
            
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