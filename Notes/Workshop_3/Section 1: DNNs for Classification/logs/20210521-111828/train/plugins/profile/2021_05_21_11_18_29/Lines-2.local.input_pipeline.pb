	\???(\??\???(\??!\???(\??	7?A?0A@7?A?0A@!7?A?0A@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\???(\??J+???A??ʡE??Y9??v????*	     8?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????![<?œ[F@)????Mb??1?M!?D@:Preprocessing2U
Iterator::Model::ParallelMapV2o??ʡ??!z????z;@)o??ʡ??1z????z;@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??ʡE??!JԮD?J,@)??(\?µ?1??F:l?+@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???(\???!?[<?œ7@)V-???1???-??"@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch;?O??n??!kW?v%j@);?O??n??1kW?v%j@:Preprocessing2F
Iterator::Model?????K??!??~Y??=@)9??v????1鰑? @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!鰑? @)?~j?t???18??18??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mb`?!?M!???)????Mb`?1?M!???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::TensorSlice????MbP?!?M!???)????MbP?1?M!???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!?M!???)????MbP?1?M!???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s8.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.97?A?0A@I?4_?gP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J+???J+???!J+???      ??!       "      ??!       *      ??!       2	??ʡE????ʡE??!??ʡE??:      ??!       B      ??!       J	9??v????9??v????!9??v????R      ??!       Z	9??v????9??v????!9??v????b      ??!       JCPU_ONLYY7?A?0A@b q?4_?gP@