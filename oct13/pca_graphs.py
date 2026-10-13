def get_scree_plot( pca_instance ) :
  variance = pca_instance.explained_variance_ratio_

  plt.figure( figsize=(10,5) )
  plt.plot( range( 1, len( variance ) + 1 ), variance, marker = "o"  )

  plt.xlabel( "Principal Component" )
  plt.ylabel( "Explained Variance Ratio" )
  plt.title( "Full ScreePlot - Titanic Dataset" )
  plt.grid(True)
  plt.show()

def get_cumulative_scree_plot( pca_instance ) :
  variance = pca_instance.explained_variance_ratio_
  cumulative_variance = np.cumsum( variance )

  plt.figure( figsize=(10,5) )
  plt.plot( range( 1, len( cumulative_variance ) + 1 ), cumulative_variance, marker = "o"  )

  plt.axhline( 0.90, linestyle="--", label="90%", color="red" )
  plt.axhline( 0.95, linestyle="--", label="95%", color="red" )
  plt.axhline( 0.99, linestyle="--", label="99%", color="red" )

  plt.xlabel( "Number of Principal Components" )
  plt.ylabel( "Cumulative Explained Variance" )
  plt.title( "C.E.V. - Titanic Dataset" )
  plt.grid(True)
  plt.legend()
  plt.show()

def obtain_pca_accuracy_at( n_components ) :
  pca = PCA( )
  X_pca = pca.fit_transform( df )
  X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split( X_pca, df_target, test_size = 0.2 )
  model.fit( X_pca_train,  y_pca_train )
  y_pca_pred = model.predict( X_pca_test )
  pca_accuracy = accuracy_score( y_pca_test, y_pca_pred )
  #print( f"accuracy score at n_components = {n_components} is { pca_accuracy * 100 }%" )


  plt.pie(
    [ pca_accuracy*100, 100*(1-pca_accuracy) ],
    labels=['matched', 'mismatched'],
    autopct='%1.2f%%',
    colors=['blue', 'red'],
  )

  plt.title(f'accuracy distribution at n_components = { n_components } ', fontsize=7, fontweight='bold')
  plt.show()

  #return pca_accuracy
