#%%

from tree_sitter import Language, Parser

#%%

Language.build_library(
  # Store the library in the `build` directory
  'parser/my-languages.so',

  # Include one or more languages
  [
    'vendor/tree-sitter-go',
    'vendor/tree-sitter-javascript',
    'vendor/tree-sitter-python',
    'vendor/tree-sitter-c',
    'vendor/tree-sitter-cpp',
    'vendor/tree-sitter-java',
    'vendor/tree-sitter-ruby',
    'vendor/tree-sitter-c-sharp',
    'vendor/tree-sitter-php'
  ]
)

#%%

lang = Language('/home/oathaha/data_of_m1/code-review-automation-experiment/script/result_evaluation/CodeBLEU/parser/my-languages.so', 'java')

# %%

parser = Parser()
parser.set_language(lang)
# %%
