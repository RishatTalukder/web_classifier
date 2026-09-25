import React, { useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'

const App = () => {
  const [loading, setLoading] = React.useState(true)
  const [model, setModel] = React.useState<tf.LayersModel | null>(null)

  useEffect(()=>{
    const loadModel = async ()=>{
      try{
        const loadedModel = await tf.loadLayersModel('model/model.json')
        
        tf.tidy(()=>{
          loadedModel.predict(tf.zeros([1,28,28,1]))
        })

        setModel(loadedModel)
        setLoading(false)
        console.log('model loaded')
      } catch(e) {
        console.error(e)
      }
    }
    loadModel()
  }, [])


  return (
    <div>
      <h1>
        Life is a Race
      </h1>
      {
        loading
      }
    </div>
  )
}

export default App