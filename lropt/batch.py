import torch
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.index import index
from cvxpy.expressions.expression import Expression
from torch import Tensor

"""
This file contains all the new batch-supporting atoms. Such atom must have the following static
methods (see the examples below):

1. "inner_batchify" that checks if an expression should be batchified, and if so, returns a new
expression using the new batched atom.

2. "get_args" that given an expression, produces a list called args that would be the inputs to
the constructor.
"""

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
    
    def torch_numeric(self, values: list[Tensor]):
        return self.elementwise_dotproduct(values[0], values[1])
    
    def elementwise_dotproduct(self, *args) -> Tensor:
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
    def inner_batchify(expr: Expression) -> Expression:
        """
        This method returns a new expression where matrix multiplications are replaced with
        elementwise dot products if the args are vectors.
        """

        def _should_transform(expr: Expression):
            """
            This is a helper function that checks if this expression needs to be transformed
            """
            
            #Check if dot-product
            op_name = getattr(expr, "OP_NAME", None)
            return (op_name=="@") and (type(expr) != multiply)

        #Change this expression if necessary
        if not _should_transform(expr):
            return expr
        return ElementwiseDotProduct(*expr.args)
    
    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the ElementwiseDotProduct atom.
        """
        return (expr)

class BatchedIndex(index):
    """
    This is an index (slicing) atom that supports batched data.
    """

    def __init__(self, expr, key, _orig_key, *args):
        super().__init__(expr, key, _orig_key, *args)
        self._orig_shape = self.args[0].shape

    def numeric(self, values):
        """
        Elementwise dot product
        """
        raise NotImplementedError("Numeric for ElementwiseIndex should not be used.")
    
    def torch_numeric(self, values: list[Tensor]):
        def _is_batch(self, values: list[Tensor]) -> bool:
            """
            This is a helper function that returns True if this is batch mode
            """
            curr_shape = len(values[0].shape)
            return (curr_shape - len(self._orig_shape)) == 1
        
        def _create_slice(key: None | int | tuple | slice) -> tuple:
            """
            This is a helper function that adds a slice(None, None, None) to key to select all the
            elements of the 0-th element (which is the batch).
            """
            #Slice to add to the key, to take all the batched elements
            add_slice = slice(None, None, None)
            #If key is a single slice, convert it to a list (will be used later).
            if isinstance(key, slice):
                key = [key]
            if key is None:
                return add_slice
            elif isinstance(key, int):
                return (add_slice, key)
            else:
                key = list(key)
                key.insert(0, add_slice)
                return tuple(key)

        batch_flag = _is_batch(self, values)
        # updated_key = _update_key(self, values, batch_flag)
        key = self._orig_key
        if batch_flag:
            #In batch mode, need to specify to take all the elements from the first dimension
            key = _create_slice(key)
        return values[0][key]
    
    @staticmethod
    def inner_batchify(expr: Expression) -> Expression:
        """
        This method returns a new expression where index (slicing) opertaions are replaced with
        batched indexing.
        """

        def _should_transform(expr: Expression):
            """
            This is a helper function that checks if this expression needs to be transformed
            """
            
            return isinstance(expr, index)

        #Change this expression if necessary
        if not _should_transform(expr):
            return expr
        return BatchedIndex(*BatchedIndex.get_args(expr))
    
    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the ElementwiseDotProduct atom.
        """
        return (expr.args[0], expr.key, expr._orig_key)

def batchify(expr: Expression) -> Expression:
    """
    This method returns a new expression where objects of target_class are transformed to 
    """
    
    #Recursively change all the args of this expression
    args = [batchify(arg) for arg in expr.args]
    #DEBUG
    # args = []
    # for arg in expr.args:
    #     args.append(batchify(arg))
    expr.args = args
    
    #Change this expression if necessary
    batched_type = SUPPORT_BATCH[type(expr)] if type(expr) in SUPPORT_BATCH else None

    if not batched_type:
        return expr
    
    return batched_type.inner_batchify(expr)

#This dictionary contains all the supported batched elements.
SUPPORT_BATCH = {
                MulExpression:  ElementwiseDotProduct,
                index:          BatchedIndex,
                }