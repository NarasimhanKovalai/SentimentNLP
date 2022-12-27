from pydantic import BaseModel

class AirlineFeedback(BaseModel):
    """A class to help create the 'predict' API endpoint 
    by taking string text as input.  """
    text:list