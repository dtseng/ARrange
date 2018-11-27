.. ARrange documentation master file, created by
   sphinx-quickstart on Sat Nov 24 20:18:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ARrange's documentation!
===================================

The config file is the main location where you can specify your preferences for your layout preferences. Objects in the layout are separated into classes -- for exampe, "person" and "item." To designate constraints for each type of object, your config file will look similar to the following:

classes:
  - .person:
    - padding: 0.001, 0.001, 0.001, 0.001
    - facing: .item # @table
    - binding: $Sittable

  - .item:
    - padding: 0.001, 0.001, 0.001, 0.001 
    - direction: <1, 0>
    - binding: $Table

Note that a period (".") before a word designates the word as a reference to a class name. (E.g., ".person" and ".item" in the example above). The list underneath each class name is a list of constraints that will be applied to each class of objects.

Below is a list of possible constraints you can use for now: 

Constraints
===================================
*padding*: left, right, top, bottom (in meters) 
    Example: padding: 0.001, 0.001, 0.001, 0.001 
    Amount of padding you want on each side of the object's bounding box. 

*binding*: surface
    Binds the object to the given type of surface. For example, if you would like the person class to only be placed on top "Sittable" surfaces, then you would add the following constraint to .person: 
        binding: $Sittable

*direction*: <x, y>
    Example: direction: <1, 0>
    Restricts object to face this direction. 

*facing*: .class_name
    Makes the object face another object of a given class name. For example, to make a person face an object of the item class, you would add the following constraint to .person: 
        facing: .item 
        



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`