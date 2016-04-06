1. Compartments with special properties (e.g. capacitance) will be modeled as "subclasses"
   - "Subclass" means a Compartment that inherits another Compartment, as is interchangeable
      with its parent in all other situations where its parent would normally be required.
   - Calcium or other ionic compartments are functionally identical (insofar as Myriad's)
     object system is concerned.
2. Section (Abstraction) Implementation
   - Sections represent collections of Compartments that concretely linked via adjacency mechs
`
class Section(object):
    
`
