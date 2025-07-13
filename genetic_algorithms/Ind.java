
import java.util.List;

public interface Ind {

    public List<Ind> recombinar(Ind ind);

    public Ind mutar();

    public double getAvaliacao();
}

