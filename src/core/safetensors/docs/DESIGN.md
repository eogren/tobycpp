Tensor Design Overview

- Tensors are pointers into a Storage block, and contain some metadata:
  - Shape std::vector<std::uint32_t>. If the shape is len(0) it's a scalar
  - Type (for now float32)

- When loading tensors there is an alignment requirement based on type and Storage type

- Most basic operations for now:
  - get(std::vector<std::uint32_t>) -- get an element of a tensor by coordinates. coords must be sizeof shape.
  - add(other vector)


Decisions:
 - Should we templatize vectors by shape? Eg Tensor<2, 3> to represent a 2x3 vector.

   Thinking: no, this is too hard to represent in c++ type system

 - Should we have types like Float32Tensor<>, Int32Tensor<> so they can have specialized get() methods that
   automatically return the right type? And if so, what's the method to get the right one? Would we need some sort
   of generic Tensor class also with .get_as_float32(), .get_as_int32() etc? [and they throw if you pass the wrong type]

TODOs:
  -
