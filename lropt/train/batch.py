"""
This file contains all the new batch-supporting atoms. Such atom must have the following
methods (see the examples below):

1. "transform" (static method) that checks if an expression should be batchified, and if so,
returns a new expression using the new batched atom.

2. "get_args" (static method) that given an expression, produces a tuple called args that would be
the inputs to the constructor.

3. "torch_numeric" that determines the actual torch numeric functionality of the atom.
"""

from functools import partial

import torch
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.expressions.expression import Expression
from torch import Tensor

from lropt.train.utils import (
    already_padded,
    expand_tensor,
    get_batch_size,
    inner_transform,
    is_batch,
    recursive_apply,
    replace_partial_args,
    stack_tensor,
)


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
            for i, batch_flag in enumerate(is_batch_vec):
                if batch_flag is None:
                    batched_args.append(expand_tensor(args[i], batch_size))
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
    def transform(expr: Expression) -> Expression:
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
            return (op_name=="@") and (not isinstance(expr, multiply))

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

    def torch_numeric(self, values: list[Tensor]):
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

        batch_flag = is_batch(self, values)
        key = self._orig_key
        if batch_flag:
            #In batch mode, need to specify to take all the elements from the first dimension
            key = _create_slice(key)
        return values[0][key]

    @staticmethod
    def transform(expr: Expression) -> Expression:
        """
        This method returns a new expression where index (slicing) opertaions are replaced with
        batched indexing.
        """
        return inner_transform(expr, BatchedIndex)

    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the BatchedIndex atom.
        """
        return (expr.args[0], expr.key, expr._orig_key)
    
class BatchedReshape(reshape):
    """
    This is a reshaping atom that supports batched data. The first dimension is assumed to be the 
    batched dimension (if batched data are given).
    """

    def __init__(self, expr, shape, *args):
        super().__init__(expr, shape, *args)
        self._orig_shape = self.args[0].shape

    def torch_numeric(self, values: list[Tensor]):
        def _calc_shape(self, values: list[Tensor], batch_flag: bool) -> list:
            """
            This is a helper function that calculates the desired shape, taking into account
            the batched reshaping.
            """
            if batch_flag and not already_padded(self.args[0], values[0]):
                shape = [values[0].shape[0]] + list(self.shape)
            else:
                shape = self.shape
            return shape
        
        def _apply_batched_reshape(self, values: list[Tensor]) -> Tensor:
            """
            This is a helper function that reshapes a tensor.
            """
            if values[0].is_sparse:
                values[0] = values[0].to_dense()
            #In batch mode, split the tensor into batches.
            batch_size = get_batch_size(self, values[0])
            #Need to break this to not add unnecessary dimensions to un-batched tensors
            if batch_size>1:
                #If batch mode, split the input tensor into batches
                values = torch.split(values[0], [1 for _ in range(batch_size)])
                res = torch.stack([super(BatchedReshape, self).torch_numeric(value) \
                                                    for value in values])
            else:
                res = super().torch_numeric(values)
            return res
        
        batch_flag = is_batch(self, values)
        orig_shape = self._shape #Keep the original shape
        #In batch mode, need to reshape but keep the batch dimension (0-th element of values[0])
        shape = _calc_shape(self, values, batch_flag)
        self._shape = shape #Update the atom's shape
        res = _apply_batched_reshape(self, values)
        self._shape = orig_shape
        return res
    
    @staticmethod
    def transform(expr: Expression) -> Expression:
        """
        This method returns a new expression where reshaping opertaions are replaced with
        batched reshaping.
        """
        return inner_transform(expr, BatchedReshape)

    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the BatchedReshape atom.
        """
        return (expr.args[0], expr.shape, expr.order)

class BatchedHstack(Hstack):
    """
    This is a Hstack atom that supports batched data. The first dimension is assumed to be the 
    batched dimension (if batched data are given).
    """

    def __init__(self, *args):
        super().__init__(*args)

    def torch_numeric(self, values: list[Tensor]) -> Tensor:
        return torch_numeric_stack(self, values)
    
    @staticmethod
    def transform(expr: Expression) -> Expression:
        """
        This method returns a new expression where reshaping opertaions are replaced with
        batched reshaping.
        """
        return inner_transform(expr, BatchedHstack)

    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the BatchedHstack atom.
        """
        return expr.args

class BatchedVstack(Vstack):
    """
    This is a Vstack atom that supports batched data. The first dimension is assumed to be the 
    batched dimension (if batched data are given).
    """

    def __init__(self, *args):
        super().__init__(*args)

    def torch_numeric(self, values: list[Tensor]) -> Tensor:
        return torch_numeric_stack(self, values)
    
    @staticmethod
    def transform(expr: Expression) -> Expression:
        """
        This method returns a new expression where reshaping opertaions are replaced with
        batched reshaping.
        """
        return inner_transform(expr, BatchedVstack)

    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the BatchedVstack atom.
        """
        return expr.args

class BatchedAddExpression(AddExpression):
    """
    This is an AddExpression atom that supports sparse matrix addition. It supports any combination
    of sparse/dense tensor added to a sparse/dense tensor.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def _first_iteration(self, values: list[Tensor]) -> bool:
        """
        This is a helper function that checks if this is the first iteration torch_numeric is called
        or not.
        """
        return len(self.args) >= len(values)

    def torch_numeric(self, values: list[Tensor]) -> Tensor:
        def _pad_tensors(self, values: list[Tensor]) -> int:
            """
            This helper function pads the tensors so they match.
            It changes the list values in place.
            Returns the batch size.
            """
            #Should happen only in the first call of torch_numeric
            batch_size = None
            if not self._first_iteration(values):
                return
            
            unpadded_inds = []
            for i, value in enumerate(values):
                if already_padded(self.args[i], value):
                    new_batch_size = get_batch_size(self.args[i], value)
                    if batch_size is None:
                        batch_size = new_batch_size
                    elif batch_size != new_batch_size:
                        raise ValueError(f"Inconsistent batch sizes found: {batch_size} and "
                                        f"{new_batch_size}")
                else:
                    unpadded_inds.append(i)
            
            #Don't pad if nothing has been padded
            if batch_size is None:
                return batch_size
            
            #Pad all unpadded tensors
            for i in unpadded_inds:
                values[i] = stack_tensor(values[i], batch_size)

            return batch_size

        def _expand_tensors(self, values: list[Tensor], batch_size: int | None) -> None:
            """
            This helper function expands all tensors so they match self.shape or
            [batch_size, self.shape].
            Updates values in place.
            """
            if not self._first_iteration(values):
                return

            if batch_size is None:
                output_shape = list(self.shape)
            else:
                output_shape = [batch_size] + list(self.shape)
            output_shape_ndim = len(output_shape)
            for i, value in enumerate(values):
                curr_ndim = value.ndim
                if curr_ndim > output_shape_ndim:
                    raise ValueError("Input tensor has more dimensions than needed")
                while curr_ndim < output_shape_ndim:
                    value = value.unsqueeze(-1)
                    if value.ndim<=curr_ndim:
                        raise ValueError("expand_tensors failed to expand the dimensions.")
                    curr_ndim = value.ndim
                values[i] = value

        def _order_inputs(values: list[Tensor]) -> list[Tensor]:
            """
            This function generates the LHS and RHS of the addition.
            It is needed because sparse + dense is not allowed in Pytorch.
            """
            if values[0].is_sparse and not values[1].is_sparse:
                lhs = values[1]
                rhs = values[0]
            else:
                lhs = values[0]
                rhs = values[1]
            return lhs, rhs
        
        def _stack_for_broadcast(lhs: Tensor, rhs: Tensor) -> list[Tensor]:
            """
            This helper function stacks the unbatched input to match the dimension of the batched
            input.
            It is needed because currently Pytorch (v2.3.0) does not support broadcasting between
            sparse and dense matrices.
            """
            while lhs.ndim > rhs.ndim:
                rhs = stack_tensor(rhs, lhs.shape[0])
            while lhs.ndim < rhs.ndim:
                lhs = stack_tensor(lhs, rhs.shape[0])
            return lhs, rhs     

        if self._first_iteration(values):
            #Pad all unpadded tensors.
            batch_size = _pad_tensors(self, values)
            #unsqueeze all tensors to agree with self.shape
            _expand_tensors(self, values, batch_size)

        #This atom does not support adding more than 2 elements directly.
        if len(values) <= 1:
            return super(BatchedAddExpression, self).numeric(values)
        else:
            rhs = self.torch_numeric(values[1:])
            values = [values[0], rhs]
        lhs, rhs = _order_inputs(values)
        #Execute the following as a try-catch block, because currently Pytorch does not support
        #sparse-dense broadcasting, but this might change in later versions.
        try:
            return super(BatchedAddExpression, self).numeric([lhs, rhs])
        except RuntimeError:
            #Stack the tensors and retry
            lhs, rhs = _stack_for_broadcast(lhs, rhs)
            return super(BatchedAddExpression, self).numeric([lhs, rhs])

    @staticmethod
    def transform(expr: Expression) -> Expression:
        """
        This method returns a new expression where additions are replaced with additions that
        support sparse-by-dense matrix (not supported by default by Pytorch)
        """
        return inner_transform(expr, BatchedAddExpression)
    
    @staticmethod
    def get_args(expr: Expression) -> tuple:
        """
        This is a helper function that returns the requires args for the SparseAddExpression atom.
        """
        return (expr.args, )

def batchify(expr: Expression | partial) -> Expression:
    """
    This function returns a new expression where atoms of expr are replaced with their batched
    counterparts.

    Parameters:
        expr (Expression or partial):
            A CVXPY Expression or Torch expression to batchify.
    
    Returns:
        expr (Expression or partial):
            A CVXPY Expression where atoms in SUPPORT_BATCH are replaced with their batched
            counterparts.

            OR

            A torch expression (partial) if a torch expression is passed.
    """
    def _replace_torch_expr_args(expr: partial, new_expr: Expression) -> partial:
        """
        This helper function replaces a torch expressions CVXPY expression with a new expression.
        """
        new_args = list(expr.args)
        new_args[0] = new_expr
        new_args = tuple(new_args)
        return replace_partial_args(expr, new_args)
    
    #Partial means a torch expression is passed
    if isinstance(expr, partial):
        batchified_expr = recursive_apply(expr.args[1], SUPPORT_BATCH) #expr.args[1] is Expression
        return _replace_torch_expr_args(expr, batchified_expr)
    return recursive_apply(expr, SUPPORT_BATCH)

#This dictionary contains all the supported batched elements.
SUPPORT_BATCH = {
                MulExpression:  ElementwiseDotProduct,
                index:          BatchedIndex,
                reshape:        BatchedReshape,
                Hstack:         BatchedHstack,
                Vstack:         BatchedVstack,
                AddExpression:  BatchedAddExpression,
                }


def permute_tensors(values: list[Tensor], forward: bool) -> None:
    """
    This helper function permutes each tensor in values:
        If forward==True, the 0-th dimension (which corresponds to the batch dimension) is moved to
        the last axis.
        If forward==False, the last element (which corresponds to the batch dimension) is moved to
        the 0-th axis. This happens after calling this function in forward mode.
    """
    def _calc_dims(value: Tensor, forward: bool):
        """
        This is a helper function that calculates the dimensions permute.
        """
        def _listify(x: list | int | float) -> list:
            """
            This is a helper function that turns x to a list with this x if x is a scalar, or
            returns the x if x is already a list.
            """
            if isinstance(x, list):
                return x
            return [x]
        dims = list(range(len(value.shape)))
        if forward:
            dims = _listify(dims[1:]) + _listify(dims[0])
        else:
            dims = _listify(dims[-1]) + _listify(dims[:-1])
        return dims
    
    for i,value in enumerate(values):
        dims = _calc_dims(value, forward)
        values[i] = torch.permute(value, dims)
            
def torch_numeric_stack(self: BatchedHstack | BatchedVstack, values: list[Tensor]) -> Tensor:
    """
    This function implements torch_numeric for either BatchedHstack or BatchedVstack.
    """
    def _check_batch(self: BatchedHstack, values: list[Tensor]) -> list[bool]:
        """
        This is an inner function that checks which elements of values are batched.
        Returns a list of length len(values), with True for elements that are batched.
        Also returns the size of the batch.
        """
        n = len(values)
        is_batch_vec = [False]*n
        batch_size = None
        curr_batch_size = None
        for i in range(n):
            is_batch_vec[i] = (len(values[i].shape) > len(self.args[i].shape))
            if is_batch_vec[i]:
                curr_batch_size = values[i].shape[0]
                #if batched_size was not set - set it
                if batch_size is None:
                    batch_size = curr_batch_size
                #if the current batch size does not agree with previously found batch sizes
                elif curr_batch_size != batch_size:
                    raise ValueError(f"Hstack was given different batch sizes: {batch_size} and"
                                        f" {curr_batch_size}.")
        return is_batch_vec, curr_batch_size
    
    def _duplicate_tensors(values: list[Tensor], is_batch_vec: list[bool],
                            batch_size: int) -> None:
        """
        This is a helper function that creates a new axis for batch and duplicates it batch_size
        times.
        """

        for i in range(len(values)):
            if is_batch_vec[i]: #Nothing to do on already-batched elements
                continue
            new_value = values[i].unsqueeze(0)
            repeat_dims = [1]*len(new_value.shape)
            repeat_dims[0] = batch_size
            values[i] = new_value.repeat(repeat_dims)

    def _pad_batched_vectors(self: BatchedHstack, values: list[Tensor]) -> None:
        """
        This is a helper function that pads a new dimension to all batched vectors.
        It is required since Hstack and Vstack work differently on vectors and tensors.
        """
        def _is_batched_vec(arg: Expression, value: Tensor) -> bool:
            """
            This is a helper function that returns True if an an arg is a batched vector.
            """
            if not is_batch(arg, [value], orig_shape_flag=False):
                return False
            return len(arg.shape)<=1 #<=1 for vectors and scalars

        for i, val in enumerate(values):
            if _is_batched_vec(self.args[i], val) and (not already_padded(self.args[i], val)):
                values[i] = val.unsqueeze(0)

    #Check which element is batched
    is_batch_vec, batch_size = _check_batch(self, values)
    
    #If none are batched, normal stack and return
    if batch_size is None:
        return super(type(self), self).torch_numeric(values)
    
    #Duplicate all non-batched elements (e.g. constants)
    _duplicate_tensors(values, is_batch_vec, batch_size)
    
    #Unsqueeze all batched vectors in the 0-th dimension
    _pad_batched_vectors(self, values)

    #Originally I did this, I'm not sure why the last element needs to be the batched one?
    #Might be BatchedVstack thing.
    # #Permute such that the batch is the last element
    # permute_tensors(values, forward=True)

    #Apply torch_numeric
    res = super(type(self), self).torch_numeric(values)
        
    # #Re-permute the inputs (not sure if needed)
    # permute_tensors(values, forward=False)
        
    #Return the values of torch_numeric
    return res

