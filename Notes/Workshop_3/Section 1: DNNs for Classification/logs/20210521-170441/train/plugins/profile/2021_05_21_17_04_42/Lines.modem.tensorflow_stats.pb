"?p
BHostIDLE"IDLE1     (?@A     (?@a??????i???????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?_@9     ?_@A     ?_@I     ?_@a|?:E??i????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      L@9      L@A     ?D@I     ?D@a?\b???iK	4?N???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      =@9      =@A      =@I      =@a(?????z?i?L???????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      9@9      9@A      9@I      9@a? ?d;w?i??v i????Unknown
aHostCast"sequential/Cast(1      5@9      5@A      5@I      5@a?R|?s?iP???o????Unknown
dHostDataset"Iterator::Model(1     ?a@9     ?a@A      0@I      0@a?q}?*?m?i?rO#,????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      .@9      .@A      .@I      .@a???h?k?i]b????Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      0@9      0@A      .@I      .@a???h?k?i??t??.???Unknown
v
HostMul"%binary_crossentropy/logistic_loss/mul(1      (@9      (@A      (@I      (@a_? Mf?i|?:E???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a%g???b?it?:??W???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      $@9      $@A      $@I      $@a%g???b?iۘ?Hej???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      $@9      $@A      $@I      $@a%g???b?iB????|???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      $@9      $@A      $@I      $@a%g???b?i??_~?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?&ع`?i9?jVJ????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?q}?*?]?i??k(????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a?q}?*?]?i?Y)?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1       @9       @A       @I       @a?q}?*?]?id????????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?q}?*?]?i????????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1       @9       @A       @I       @a?q}?*?]?i?G??????Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1       @9       @A       @I       @a?q}?*?]?i?T??~????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?q}?*?]?iH??\???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?íf?Z?i*??>_???Unknown
VHostMean"Mean(1      @9      @A      @I      @a?íf?Z?iAl?a"???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a?íf?Z?i???c/???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a?íf?Z?i???6f<???Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      @9      @A      @I      @a?íf?Z?i?E??hI???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a_? MV?i????T???Unknown
jHostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a_? MV?i?#???_???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a_? MV?iӒ?9?j???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a%g???R?i?'t???Unknown
j HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a%g???R?i;?S?q}???Unknown
v!HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a%g???R?io(???????Unknown
?"HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a%g???R?i??
o????Unknown
y#HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a%g???R?i?6f<R????Unknown
?$HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a%g???R?i??	?????Unknown
?%HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a%g???R?i?E??????Unknown
o&HostSigmoid"sequential/dense_3/Sigmoid(1      @9      @A      @I      @a%g???R?is?x?2????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?q}?*?M?i?k(??????Unknown
e(Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?q}?*?M?i+ع????Unknown?
?)HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?q}?*?M?i????????Unknown
?*HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?q}?*?M?i?I7??????Unknown
V+HostSum"Sum_2(1      @9      @A      @I      @a?q}?*?M?i????]????Unknown
~,HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?q}?*?M?i?????????Unknown
v-HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?q}?*?M?i?'F?;????Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?q}?*?M?iS????????Unknown
?/HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?q}?*?M?i?f?????Unknown
?0HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?q}?*?M?iU?????Unknown
?1Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?q}?*?M?ig?????Unknown
?2HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?q}?*?M?i?D?$g???Unknown
?3HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?q}?*?M?i?c/????Unknown
?4HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?q}?*?M?i{?:E???Unknown
?5HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a?q}?*?M?i?"?D?$???Unknown
?6HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a?q}?*?M?i3?rO#,???Unknown
t7HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a_? MF?i?yv??1???Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a_? MF?i=1z?I7???Unknown
\9HostGreater"Greater(1      @9      @A      @I      @a_? MF?i??}'?<???Unknown
s:HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a_? MF?iG??opB???Unknown
u;HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a_? MF?i?W??H???Unknown
?<HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a_? MF?iQ???M???Unknown
z=HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a_? MF?i?ƌG*S???Unknown
|>HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a_? MF?i[~???X???Unknown
`?HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a_? MF?i?5??P^???Unknown
~@HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a_? MF?ie???c???Unknown
?AHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a_? MF?iꤛgwi???Unknown
?BHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a_? MF?io\??
o???Unknown
qCHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a_? MF?i????t???Unknown
tDHost_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @a_? MF?iy˦?1z???Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?q}?*?=?i'????}???Unknown
VFHostCast"Cast(1       @9       @A       @I       @a?q}?*?=?i?jVJ?????Unknown
XGHostCast"Cast_3(1       @9       @A       @I       @a?q}?*?=?i?:??W????Unknown
XHHostCast"Cast_4(1       @9       @A       @I       @a?q}?*?=?i1
U????Unknown
XIHostEqual"Equal(1       @9       @A       @I       @a?q}?*?=?i??]?ƌ???Unknown
`JHostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?q}?*?=?i???_~????Unknown
?KHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      M@9      M@A       @I       @a?q}?*?=?i;y?5????Unknown
TLHostMul"Mul(1       @9       @A       @I       @a?q}?*?=?i?Hej?????Unknown
|MHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?q}?*?=?i??亂???Unknown
dNHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?q}?*?=?iE?u\????Unknown
vOHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?q}?*?=?i??l?????Unknown
vPHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?q}?*?=?i???˦???Unknown
?QHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?q}?*?=?iOW?????Unknown
}RHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?q}?*?=?i?&t?:????Unknown
uSHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?q}?*?=?i????????Unknown
bTHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?q}?*?=?iY?#??????Unknown
wUHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?q}?*?=?i?{a????Unknown
xVHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?q}?*?=?i?eӟ????Unknown
?WHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?q}?*?=?ic5+%?????Unknown
?XHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?q}?*?=?i???????Unknown
~YHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?q}?*?=?i???/?????Unknown
?ZHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?q}?*?=?im?2??????Unknown
}[HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?q}?*?=?it?:?????Unknown
\HostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1       @9       @A       @I       @a?q}?*?=?i?C??e????Unknown
?]HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?q}?*?=?iw:E????Unknown
?^HostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?q}?*?=?i%????????Unknown
v_HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?q}?*?-?i??=??????Unknown
X`HostCast"Cast_5(1      ??9      ??A      ??I      ??a?q}?*?-?iӲ?O?????Unknown
aaHostIdentity"Identity(1      ??9      ??A      ??I      ??a?q}?*?-?i???h????Unknown?
?bHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?q}?*?-?i??A?C????Unknown
rcHostAdd"!binary_crossentropy/logistic_loss(1      ??9      ??A      ??I      ??a?q}?*?-?iXj??????Unknown
wdHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?q}?*?-?i/R?Z?????Unknown
yeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?q}?*?-?i:E?????Unknown
?fHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?q}?*?-?i?!?߲????Unknown
?gHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?q}?*?-?i?	???????Unknown
?hHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?q}?*?-?i??Hej????Unknown
?iHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?q}?*?-?ib??'F????Unknown
?jHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?q}?*?-?i9???!????Unknown
?kHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?q}?*?-?i?L??????Unknown
?lHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?q}?*?-?i???o?????Unknown
?mHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?q}?*?-?i?x?2?????Unknown
?nHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?q}?*?-?i?`P??????Unknown
?oHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?q}?*?-?ilH??l????Unknown
pHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a?q}?*?-?iC0?zH????Unknown
?qHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?q}?*?-?iT=$????Unknown
?rHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?q}?*?-?i?????????Unknown
YsHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?o
sHostDataset"Iterator::Model::ParallelMapV2(1     ?_@9     ?_@A     ?_@I     ?_@a???????i????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      L@9      L@A     ?D@I     ?D@a?5???i5?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      =@9      =@A      =@I      =@a??1G????i"???c????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      9@9      9@A      9@I      9@a??WV???i?N??N????Unknown
aHostCast"sequential/Cast(1      5@9      5@A      5@I      5@a??
цϟ?in??1G????Unknown
dHostDataset"Iterator::Model(1     ?a@9     ?a@A      0@I      0@aݾ?z?<??i\$??m???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      .@9      .@A      .@I      .@a??θ??i??؉?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      0@9      0@A      .@I      .@a??θ??i?	j*D???Unknown
v	HostMul"%binary_crossentropy/logistic_loss/mul(1      (@9      (@A      (@I      (@a&?q-??i?wɃg???Unknown
i
HostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a?n_Y?K??i!s?n_Y???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      $@9      $@A      $@I      $@a?n_Y?K??i?n_Y?K???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      $@9      $@A      $@I      $@a?n_Y?K??ij*D>???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      $@9      $@A      $@I      $@a?n_Y?K??i??z?<???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?	j*D??i?"AM????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @aݾ?z?<??i???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @aݾ?z?<??im??1G???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1       @9       @A       @I       @aݾ?z?<??i8?Z$????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aݾ?z?<??iθ	???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1       @9       @A       @I       @aݾ?z?<??iθ	j???Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1       @9       @A       @I       @aݾ?z?<??i???t?????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @aݾ?z?<??i?c???+???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @ag\?5??i?ջ??????Unknown
VHostMean"Mean(1      @9      @A      @I      @ag\?5??i0G???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @ag\?5??i̸	j*???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @ag\?5??ih*D>???Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      @9      @A      @I      @ag\?5??i?q-????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a&?q-??iA???????Unknown
jHostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a&?q-??i~?Q?}e???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a&?q-??i????3????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?n_Y?K~?i??t??????Unknown
jHostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?n_Y?K~?iub'vb'???Unknown
v HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?n_Y?K~?iR!???c???Unknown
?!HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?n_Y?K~?i/??k?????Unknown
y"HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a?n_Y?K~?i???(????Unknown
?#HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?n_Y?K~?i?]?`????Unknown
?$HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?n_Y?K~?i???WV???Unknown
o%HostSigmoid"sequential/dense_3/Sigmoid(1      @9      @A      @I      @a?n_Y?K~?i??WV?????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aݾ?z?<x?i!AM?h????Unknown
e'Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aݾ?z?<x?i??B??????Unknown?
?(HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aݾ?z?<x?i8?Z$???Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aݾ?z?<x?i?q-?T???Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @aݾ?z?<x?i?"AM????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aݾ?z?<x?i?<pƵ???Unknown
v,HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aݾ?z?<x?i???????Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aݾ?z?<x?i?θ???Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aݾ?z?<x?im??1G???Unknown
?/HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aݾ?z?<x?i???+?w???Unknown
?0Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aݾ?z?<x?i8?Z$????Unknown
?1HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aݾ?z?<x?i??؉?????Unknown
?2HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aݾ?z?<x?i	θ	???Unknown
?3HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aݾ?z?<x?i?h???9???Unknown
?4HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @aݾ?z?<x?iθ	j???Unknown
?5HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @aݾ?z?<x?i?3?E?????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a&?q-r?i???(ݾ???Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a&?q-r?i?K8????Unknown
\8HostGreater"Greater(1      @9      @A      @I      @a&?q-r?i?WV?????Unknown
s9HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a&?q-r?i?c???+???Unknown
u:HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a&?q-r?ipƵHP???Unknown
?;HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a&?q-r?i7|???t???Unknown
z<HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a&?q-r?iU?6|?????Unknown
|=HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a&?q-r?is?n_Y????Unknown
`>HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a&?q-r?i???B?????Unknown
~?HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a&?q-r?i???%???Unknown
?@HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a&?q-r?i͸	j*???Unknown
?AHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a&?q-r?i??N??N???Unknown
qBHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a&?q-r?i	ц?s???Unknown
tCHost_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @a&?q-r?i'ݾ?z????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aݾ?z?<h?i??9J?????Unknown
VEHostCast"Cast(1       @9       @A       @I       @aݾ?z?<h?i?B???????Unknown
XFHostCast"Cast_3(1       @9       @A       @I       @aݾ?z?<h?id?.y0????Unknown
XGHostCast"Cast_4(1       @9       @A       @I       @aݾ?z?<h?i#??m????Unknown
XHHostEqual"Equal(1       @9       @A       @I       @aݾ?z?<h?i?Z$?????Unknown
`IHostGatherV2"
GatherV2_1(1       @9       @A       @I       @aݾ?z?<h?i????(???Unknown
?JHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      M@9      M@A       @I       @aݾ?z?<h?i`??"A???Unknown
TKHostMul"Mul(1       @9       @A       @I       @aݾ?z?<h?is?n_Y???Unknown
|LHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aݾ?z?<h?i?%?q???Unknown
dMHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aݾ?z?<h?i?؉?؉???Unknown
vNHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aݾ?z?<h?i\?5????Unknown
vOHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aݾ?z?<h?i>?Q????Unknown
?PHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aݾ?z?<h?i???c?????Unknown
}QHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aݾ?z?<h?i??t??????Unknown
uRHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aݾ?z?<h?iXV?????Unknown
bSHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aݾ?z?<h?i	j*D???Unknown
wTHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aݾ?z?<h?iֻ???3???Unknown
xUHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aݾ?z?<h?i?n_Y?K???Unknown
?VHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @aݾ?z?<h?iT!???c???Unknown
?WHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @aݾ?z?<h?i?T?6|???Unknown
~XHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aݾ?z?<h?i҆?s????Unknown
?YHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aݾ?z?<h?i?9J??????Unknown
}ZHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aݾ?z?<h?iP??N?????Unknown
[HostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1       @9       @A       @I       @aݾ?z?<h?i???(????Unknown
?\HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aݾ?z?<h?i?Q?}e????Unknown
?]HostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aݾ?z?<h?i?5????Unknown
v^HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aݾ?z?<X?i?]?`????Unknown
X_HostCast"Cast_5(1      ??9      ??A      ??I      ??aݾ?z?<X?iK????%???Unknown
a`HostIdentity"Identity(1      ??9      ??A      ??I      ??aݾ?z?<X?i?m??1???Unknown?
?aHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??aݾ?z?<X?i	j*D>???Unknown
rbHostAdd"!binary_crossentropy/logistic_loss(1      ??9      ??A      ??I      ??aݾ?z?<X?ih???9J???Unknown
wcHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aݾ?z?<X?i???WV???Unknown
ydHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aݾ?z?<X?i&vb'vb???Unknown
?eHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aݾ?z?<X?i??s?n???Unknown
?fHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aݾ?z?<X?i?(ݾ?z???Unknown
?gHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aݾ?z?<X?iC??
ц???Unknown
?hHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??aݾ?z?<X?i??WV?????Unknown
?iHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aݾ?z?<X?i5?????Unknown
?jHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aݾ?z?<X?i`???+????Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aݾ?z?<X?i???9J????Unknown
?lHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??aݾ?z?<X?iAM?h????Unknown
?mHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aݾ?z?<X?i}?
ц????Unknown
?nHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aݾ?z?<X?i????????Unknown
oHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??aݾ?z?<X?i;M?h?????Unknown
?pHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aݾ?z?<X?i??B??????Unknown
?qHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aݾ?z?<X?i?????????Unknown
YrHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU