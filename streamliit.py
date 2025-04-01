library(magick)
library(httr)
library(jsonlite)

cohere_api_key <- "7HpPc5ghLeUjTWUHljnX8Y7xgxRcRhdOUg2bv9Px"

generate_caption <- function(image_path) {

  return("A generated caption for the uploaded image.")
}

query_cohere <- function(caption, user_question) {
  prompt <- paste("Image Caption:", caption, "\nUser Question:", user_question, "\nAnswer:")
  
  response <- POST(
    url = "https://api.cohere.ai/v1/generate",
    add_headers(Authorization = paste("Bearer", cohere_api_key), `Content-Type` = "application/json"),
    body = toJSON(list(
      model = "command-xlarge",
      prompt = prompt,
      max_tokens = 100,
      temperature = 0.75
    ), auto_unbox = TRUE)
  )
  
  content <- content(response, as = "text", encoding = "UTF-8")
  parsed_content <- fromJSON(content)
  return(parsed_content$generations[[1]]$text)
}
