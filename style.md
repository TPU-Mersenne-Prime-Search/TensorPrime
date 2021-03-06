Settings.txt is used to store and modify settings
Any string starting with a dash and split with a colon is recognized as a setting,
all other text is ignored.



Python Style Guide

Based on python's PEP 8 style guide  
<https://www.python.org/dev/peps/pep-0008/>  
<https://github.com/python/peps/blob/master/pep-0008.txt>

Dependency specification  
<https://www.python.org/dev/peps/pep-0508/>


Layout

Indentation
 - hanging indents add a level (or partial if spaces)
 - Tabs should be used, as it allows a reader to use custom tab spacing if preffered.
 - A maximum line length of 70 characters, but exceptable in cases where code would be more difficult to read if split.
 - brackets/perenthesis that are too long or nested should indent between the brackets and the items within. For multi-dimentional cases, further indenting should be added per layer.

       a = [
           12345,
           6789
       ]
    
       a = [
           [1, 2, 3],
           [4, 5, 6]
       ]

Whitespace in expressions
 - Operations should have whitespace on both sides of every operator, using parenthesis at every transition between operator precedence. (pemdas, modulo, comparation, assignment)

       x = a + (b * c / (d * e)) + f

 - Commas and semicolons have a one space after, but not before; excepting trailing spaces.
 - For multiline operations, the operator between the two lines should be placed at the beginning of the following line.
 - top level function and class definitions are surrounded with two blank lines.
   Method definitions inside a class are surrounded by one blank line.
 - Conditionals should have trailing whitespace after their code block ends.
   Closely related or extremely short functions/statements may ommit these spacings.

Comments
 - Comments should be able to be split down to 50 characters, becoming multiline if nessesary.
 - Comments should preceed the code they relate to, excepting split lines with complex interiors (long if statements) where inline comments can be used.
 - any comment that is large enough to warrent internal blank lines should use block comments.

Imports
 - imports should be done on seperate lines for each item imported
 - module level dunder names go before imports, excepting __future__ imports which must come before all other code.
 - Imports are grouped by Standard libraries, then third party, then local or library specific; with an empty line between each group.

String quotes
 - double quotes (") should be used for code, and single (') for comments

Naming conventions
 - package and module names should be short and all lowercase.
 - Class names should use CapWords
 - Exceptions take the form of classes and should use the same convention, additionally including the word Error.
 - function and variable names should be mixedCase
 - global variables use the same conventions as functions.
 - Constants are written with all capitals seperated by underscores.

programming recomendations
 - comparisons to singletons like None should always be done with "is" or "is not", never the equality operators.


