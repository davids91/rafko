Contributing
===

When Contributing, please consider that I have limited resources, so I might not be able to respond immediately. The Style with a few exceptions follows the [Google Style Guide](https://google.github.io/styleguide/). All suggestions welcome!

Rafko Style Guide
===

  - Naming best practices:
    - namespace names: `have_underscore_between_words`
    - entity names(classes, protobuf messages): `AreDoneWithCamelCase`
    - class member variables: `m_prePendedAndCamelCase`
    - static variables: `s_sameAsMembersButPrependedWithS`
    - class member functions: `start_with_lowercase_and_are_camelcase`
    - local variable names: `are_like_namespaces`
    - enumerations:
      - enumeration names: `Start_with_uppercase_but_theres_underscore`
      - enumeration members: `start_with_lowercase_but_otherwise_the_same`
    - maximum name length: `as_you_can_see_there_is_great_length_one_can_go_to_name_their_object_as_descriptive_as_possible`
  - Comments: Please only use `/* this style */`, because there might be line ending inconsistencies between different operating systems
