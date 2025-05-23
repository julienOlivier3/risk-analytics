from IPython.display import display, HTML
import numpy as np
import scipy.sparse
import torch
from typing import Union, Dict


def print_vector(
        vector: Union[np.ndarray, scipy.sparse._csr.csr_matrix],
        rounding_digit: int = 1,
        string: str = "Vector:<br> ",
        return_output: bool = False) -> str:
    """
    Print a formatted representation of a vector as a mathematical vector.

    The output includes the first three elements, the last three elements,
    and all non-zero elements of the vector, with ellipses indicating omitted
    elements. The dimension of the vector is displayed as a subscript.

    Parameters
    ----------
    vector : Union[np.ndarray, scipy.sparse._csr.csr_matrix]
        A vector to be formatted, which can be a dense NumPy array or a
        sparse matrix.
    rounding_digit : int, optional
        Number of decimal places to round to (default is 1).
    string : str, optional
        Prefix string for the output (default is "Vector:<br> ").
    return_output : bool, optional
        If True, return the formatted string instead of displaying it
        (default is False).

    Returns
    -------
    str
        The formatted vector string if return_output is True; otherwise,
        None.
    """
    # Convert the sparse matrix to a dense array
    if isinstance(vector, scipy.sparse._csr.csr_matrix):
        vector = vector.toarray().flatten()
    elif isinstance(vector, np.ndarray):
        vector = vector.flatten()
    elif isinstance(vector, torch.Tensor):
        vector = vector.cpu().numpy().flatten()

    # Round the elements to the specified number of decimal places
    rounded_vector = np.round(vector, rounding_digit)

    # Get the indices of non-zero elements
    non_zero_indices = np.nonzero(rounded_vector)[0]

    # Prepare the formatted vector
    formatted_vector = []
    dimension = rounded_vector.shape[0]

    # Add the first three elements
    formatted_vector.extend(vector[:3].round(3))

    if (len(non_zero_indices) == len(vector)) or len(non_zero_indices) == 0:
        formatted_vector.append("...")
    else:
        # Add ellipsis if there are elements between the first three and the first non-zero
        if non_zero_indices.size > 0 and non_zero_indices[0] > 3:
            formatted_vector.append("...")

        # Add non-zero elements with ellipses in between if necessary
        for i, idx in enumerate(non_zero_indices):
            if i > 0 and idx > non_zero_indices[i - 1] + 1:
                formatted_vector.append("...")
            formatted_vector.append(vector[idx].round(3))

        # Add ellipsis if there are elements between the last non-zero and the last three
        if non_zero_indices.size > 0 and non_zero_indices[-1] < dimension - 4:
            formatted_vector.append("...")

    # Add the last three elements
    formatted_vector.extend(vector[-3:].round(3))

    # Create the output string with dimension as subscript
    output = f"{string}x<sub>(1x{dimension})</sub> = [{' '.join(map(str, formatted_vector))}]"

    if return_output:
        return output
    else:
        # Display the formatted vector
        display(HTML(output))


def print_bert_vector(last_hidden_state: np.ndarray, tokens: list[str]) -> None:
    """
    Print the BERT vector in a readable format with proper HTML rendering.

    This function takes the last hidden state from a BERT model and a list
    of tokens, and prints the contextualized embeddings in a formatted
    HTML table.

    Parameters
    ----------
    last_hidden_state : np.ndarray
        The last hidden state output from the BERT model, which contains
        the contextualized embeddings for each token.
    tokens : list[str]
        A list of tokens corresponding to the input text for which the
        embeddings are generated.

    Returns
    -------
    None
    """
    contextualized_vectors = last_hidden_state.detach().numpy()[0]

    # Find the maximum word length for alignment
    max_word_length = max(len(token) for token in tokens)

    # Prepare the HTML output
    html_output = "Contextualized BERT embeddings:<table><br>"
    for i, token in enumerate(tokens):
        # Generate the vector as an HTML string
        vector_html = print_vector(contextualized_vectors[i], rounding_digit=6, string="", return_output=True)
        # Add a row to the HTML table
        html_output += f"""<tr><td>{token:<{max_word_length}} -> </td>
        <td style='text-align:left'>{vector_html}</td></tr>"""
    html_output += "</table>"

    # Display the HTML table
    display(HTML(html_output))


def print_sparse_vector(
    vector: Union[np.ndarray, scipy.sparse._csr.csr_matrix],
    vocabulary: Dict[str, int]
) -> None:
    """
    Print the non-zero elements of a sparse vector with their corresponding vocabulary words.

    This function takes a sparse vector and a vocabulary dictionary, and prints
    each non-zero element with its corresponding word and index.

    Args:
        vector (np.ndarray or scipy.sparse.csr_matrix): The input sparse vector.
        vocabulary (Dict[str, int]): A dictionary mapping words to their indices.

    Example:
        >>> vocab = {'apple': 0, 'banana': 1, 'cherry': 2}
        >>> vec = np.array([[0, 2, 0]])
        >>> print_sparse_vector(vec, vocab)
        banana : 1 	 -> 2
    """
    if isinstance(vector, scipy.sparse._csr.csr_matrix):
        vector = vector.toarray()
    if isinstance(vocabulary, list):
        vocabulary = {word: index for index, word in enumerate(vocabulary)}

    indices = np.where(vector != 0)[1]
    values = vector[0, indices]
    words = [word for word, index in vocabulary.items() if index in indices]

    # Find the maximum word length for alignment
    max_word_length = max(len(word) for word in words) if words else 0

    print_vector(vector.flatten())

    print("Non-zero elements:")
    for index, vocab_index in enumerate(indices):
        i = int(vocab_index)
        # Use f-string with left alignment to stack values
        print(f"{words[index]:<{max_word_length}} : {i:<5} \t-> {values[index]}")
